# AKG Kernel Gen — AscendAgentRL Example

Train AKG's `KernelGen` agent on [KernelBench](https://github.com/KernelBench/KernelBench)
using slime's GRPO pipeline on GPU. AKG runs as a black-box subprocess;
all LLM calls are intercepted by `ModelMonitor` and `token_ids` are recorded
for gradient computation. Training metrics are tracked via WandB.

## Prerequisites

1. **slime** (included as submodule at `third_party/slime`):
   ```bash
   cd AscendAgentRL
   git submodule update --init
   pip install -e third_party/slime
   ```

2. **Megatron-LM** (required by slime for training backend):
   ```bash
   # Ensure Megatron-LM is available; slime uses it for distributed training.
   export MEGATRON_PATH=/path/to/Megatron-LM
   ```

3. **KernelBench dataset** (PyTorch, Level 1):
   ```bash
   python examples/akg_kernel_gen/scripts/prepare_kernelbench_data.py \
       --kernelbench-dir /path/to/KernelBench/KernelBench \
       --level level1 \
       --output data/kernelbench_level1.jsonl
   ```

4. **Model checkpoint** (Qwen3-4B):
   ```bash
   # HuggingFace weights
   HF_CHECKPOINT=/path/to/Qwen3-4B

   # Convert to torch_dist format for slime training
   python third_party/slime/tools/checkpoint/hf_to_torch_dist.py \
       --hf-checkpoint $HF_CHECKPOINT \
       --output /path/to/Qwen3-4B-torch-dist

   # slime save directory (created on first save)
   SLIME_CHECKPOINT=/path/to/Qwen3-4B-slime
   ```

5. **Register a CUDA worker** with AKG's WorkerManager:
   ```python
   from akg_agents.core.worker.manager import register_local_worker
   import asyncio
   asyncio.run(register_local_worker([0], backend="cuda", arch="a100"))
   ```

6. **Install dependencies**:
   ```bash
   pip install -e /path/to/akg/akg_agents --no-build-isolation
   pip install -e .
   ```

7. **WandB** (optional but recommended):
   ```bash
   pip install wandb
   export WANDB_API_KEY="your-api-key"
   ```

## Quickstart

```bash
export WANDB_API_KEY="your-api-key"
bash examples/akg_kernel_gen/run_qwen3_4b_akg.sh
```

With custom paths:

```bash
HF_CHECKPOINT=/data1/models/Qwen/Qwen3-4B \
REF_CHECKPOINT=/data1/checkpoints/Qwen3-4B-torch-dist \
SLIME_CHECKPOINT=/data1/checkpoints/Qwen3-4B-slime \
WANDB_API_KEY="your-key" \
NUM_GPUS=8 \
bash examples/akg_kernel_gen/run_qwen3_4b_akg.sh
```

## Architecture

```
slime rollout
  └─ custom_generate (slime_generate.py)
       └─ AgentPipe.run()
            ├─ ModelMonitor (aiohttp proxy, intercepts LLM calls)
            │   └─ SlimeSglangBackend → slime router /generate
            └─ MASLauncher → akg_rl_entry.py --config /tmp/xxx.yaml --task '...'
                  ├─ KernelDesigner → LLM (model=kernel_designer) → ModelMonitor
                  ├─ KernelGen      → LLM (model=kernel_gen)      → ModelMonitor
                  └─ KernelVerifier → local GPU (no LLM)
  └─ AKGKernelRewardProvider.compute(trajectory)
       ├─ extract final code from kernel_gen[-1].response_text
       ├─ re-run KernelVerifier → r_correct
       └─ reward = α·r_correct + β·r_perf·r_correct + γ·r_iter
  └─ trajectory_to_sample (last_turn strategy)
       └─ EpisodeResult → slime Sample
```

**Why `akg_rl_entry.py` always exits 0:**
slime drops any episode with non-zero exit code before reward computation,
removing the negative examples GRPO needs. Correctness is measured by
`AKGKernelRewardProvider`, not by the subprocess exit code.

## WandB Integration

Training metrics are automatically logged to WandB when `WANDB_API_KEY` is set:

- **Rollout metrics**: reward distribution, response length, episode success rate
- **Training metrics**: loss, gradient norms, learning rate, KL divergence
- **Eval metrics**: periodic evaluation scores

Configure via environment variables or CLI:

| Variable | Default | Description |
|----------|---------|-------------|
| `WANDB_API_KEY` | (required) | WandB API key |
| `WANDB_PROJECT` | `ascendrl-akg-kernel-gen` | WandB project name |
| `WANDB_GROUP` | `qwen3-4b-kernelbench-l1` | WandB run group |

slime's `--use-wandb` flag is automatically set when `WANDB_API_KEY` is present.

## Reward Formula

```
reward = α·r_correct + β·r_perf·r_correct + γ·r_iter

r_correct  = 1.0 if KernelVerifier passes, else 0.0
r_perf     = speedup vs reference (Phase 2, disabled by default)
r_iter     = 1.0 - n_turns / max_turns

Phase 1 defaults: α=1.0  β=0.3  γ=0.1  enable_profiling=False
```

## Sample Generation Strategy

| Phase | Strategy | Description |
|-------|----------|-------------|
| 1 (MVP) | `last_turn` | Only the final `kernel_gen` turn becomes a training sample |
| 2 | `aggregate_role` | Multi-turn loss mask for `kernel_gen` across all turns |
| 3 | `multi_role` | Separate samples per role (designer + gen) |

## Hardware Requirements

| Setup | GPUs | TP | SGLang engines | Batch size |
|-------|------|----|----------------|------------|
| 8× RTX 3090 | 8 | 2 | 4 | 8 prompts × 4 samples |
| 4× A100 | 4 | 2 | 2 | 8 prompts × 4 samples |

## Directory Structure

```
examples/akg_kernel_gen/
├── README.md
├── run_qwen3_4b_akg.sh
├── configs/
│   ├── akg_config_template.yaml     # MAS subprocess config
│   └── slime_akg_kernel_gen.yaml    # slime custom config
├── mas_entry/
│   └── akg_rl_entry.py              # subprocess entry point
├── ascendrl_glue/
│   ├── akg_kernel_reward.py         # reward provider
│   ├── kernelbench_jsonl.py         # dataset loader
│   ├── slime_generate.py            # custom_generate entry
│   ├── slime_sglang_backend.py      # InferenceBackend for slime router
│   └── trajectory_to_sample.py      # EpisodeResult → slime Sample
├── scripts/
│   └── prepare_kernelbench_data.py  # KernelBench → JSONL
└── tests/
    ├── test_akg_rl_entry.py
    ├── test_akg_kernel_reward.py
    ├── test_kernelbench_jsonl.py
    └── test_trajectory_to_sample.py
```

## Differences from OrchRL Version

| Aspect | OrchRL | AscendAgentRL |
|--------|--------|---------------|
| Training framework | VeRL | slime |
| Monitoring | manual | WandB integrated |
| Rollout adapter | `MateRolloutAdapter` | slime `custom_generate` |
| Inference backend | `VLLMBackend` | `SlimeSglangBackend` |
| Trajectory engine | `orchrl.agent_trajectory_engine` | `ascend_agent_rl.agent_trajectory_engine` |

## Running Tests

```bash
cd AscendAgentRL
python -m pytest examples/akg_kernel_gen/tests/ -v
```
