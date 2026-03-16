#!/usr/bin/env bash
# Launch slime GRPO training for AKG Kernel Gen on GPU with Qwen3-4B-Instruct.
#
# Hardware: 8× RTX 3090 (24 GB each)
# Model:    Qwen3-4B-Instruct-2507
# Dataset:  KernelBench Level 1 (100 tasks)
# Framework: slime + Megatron-LM + SGLang + Ray
#
# Usage:
#   export WANDB_API_KEY="your-key"
#   bash examples/akg_kernel_gen/run_qwen3_4b_akg.sh

set -eo pipefail

# ── Cleanup stale processes ──────────────────────────────────────────────────
pkill -9 -f sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
sleep 2

set -ex

export PYTHONBUFFERED=16

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SLIME_DIR="${PROJECT_ROOT}/third_party/slime"
MEGATRON_PATH="${MEGATRON_PATH:-/data1/lmy/agentic-rl/Megatron-LM}"

# NVLink detection (grep returns 1 when no match — guard against pipefail)
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | { grep -o 'NV[0-9][0-9]*' || true; } | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ── Model config (Qwen3-4B-Instruct-2507: rope_theta=5000000) ───────────────
export MODEL_ARGS_ROTARY_BASE=5000000
source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

# ── Checkpoints ──────────────────────────────────────────────────────────────
HF_CHECKPOINT="${HF_CHECKPOINT:-/data1/models/Qwen/Qwen3-4B-Instruct-2507}"
REF_CHECKPOINT="${REF_CHECKPOINT:-/data1/lmy/agentic-rl/checkpoints/Qwen3-4B-Instruct-2507-torch-dist}"
SLIME_CHECKPOINT="${SLIME_CHECKPOINT:-/data1/lmy/agentic-rl/checkpoints/Qwen3-4B-Instruct-2507-slime}"
mkdir -p "${SLIME_CHECKPOINT}"

CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --load "${REF_CHECKPOINT}"
    --save "${SLIME_CHECKPOINT}"
    --save-interval 10
)

# ── Data ─────────────────────────────────────────────────────────────────────
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/kernelbench_level1.jsonl}"

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_DATA}"
    --input-key prompt
    --rollout-shuffle
    --num-rollout 10
    --rollout-batch-size 2
    --n-samples-per-prompt 2
    --rollout-max-response-len 1024
    --rollout-temperature 1.0
    --global-batch-size 4
    --balance-data
)

# ── Custom generate (AKG episode) ───────────────────────────────────────────
CUSTOM_ARGS=(
    --custom-generate-function-path \
        "examples.akg_kernel_gen.ascendrl_glue.slime_generate.generate_akg_episode"
)

# ── GRPO ─────────────────────────────────────────────────────────────────────
GRPO_ARGS=(
    --advantage-estimator grpo
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
)

# ── Optimizer ────────────────────────────────────────────────────────────────
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

# ── Performance (8× RTX 3090, 24 GB) ────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-8}"

PERF_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 1536
)

SGLANG_ARGS=(
    --rollout-num-gpus 4
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.80
)

# ── WandB ────────────────────────────────────────────────────────────────────
WANDB_PROJECT="${WANDB_PROJECT:-ascendrl-akg-kernel-gen}"
WANDB_GROUP="${WANDB_GROUP:-qwen3-4b-instruct-kernelbench-l1}"

if [ -n "${WANDB_API_KEY:-}" ]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project "${WANDB_PROJECT}"
        --wandb-group "${WANDB_GROUP}"
        --wandb-key "${WANDB_API_KEY}"
    )
    echo "[INFO] WandB enabled: project=${WANDB_PROJECT} group=${WANDB_GROUP}"
else
    WANDB_ARGS=()
    echo "[INFO] WandB disabled (no WANDB_API_KEY). Set WANDB_API_KEY to enable."
fi

# ── Misc ─────────────────────────────────────────────────────────────────────
MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

# ── Ray cluster ──────────────────────────────────────────────────────────────
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_TMPDIR="${RAY_TMPDIR:-/data1/lmy/ray_tmp}"
mkdir -p "${RAY_TMPDIR}"

ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --temp-dir "${RAY_TMPDIR}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_PATH}:${PROJECT_ROOT}:${SLIME_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

# ── Launch ───────────────────────────────────────────────────────────────────
cd "${SLIME_DIR}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    --working-dir "${SLIME_DIR}" \
    -- python3 train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "$@"
