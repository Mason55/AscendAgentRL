# AKG Kernel Agent 实现方案文档

## 1. 概述

本文档描述 `AscendAgentRL` 中 AKG Kernel Agent 的**已落地实现方案**。该方案将 `OrchRL/examples/akg_kernel_gen` 的多智能体内核生成样例迁移到以 `slime` 为训练后端的架构上，通过 `agent_trajectory_engine` 实现无侵入的轨迹采集，并以 GRPO 算法进行强化学习训练。

核心设计目标：

- 保持 `akg_agents` 多智能体系统（LangGraphTask）原有执行形态不变
- 复用 `agent_trajectory_engine` 做 episode 级轨迹采集
- 通过 slime 的 `custom_generate` 接口接入，不修改 slime 主干
- 第一阶段只训练 `kernel_gen`（代码生成角色），冻结 `kernel_designer`（设计角色）

## 2. 系统架构

### 2.1 整体链路

```
slime rollout loop
  │
  ▼  (async, 多 sample 并发)
generate_akg_episode()          ← slime custom_generate 入口
  │
  ├─ _ensure_pipe_state()       ← 首次调用时初始化 backend / tokenizer / config
  │
  ▼
AgentPipe.run(prompt, reward_provider)
  │
  ├─ 1. ModelMonitor.start()    ← 启动 aiohttp 服务，监听临时端口
  │     暴露 /v1/chat/completions 端点
  │
  ├─ 2. MASLauncher.prepare_config()
  │     将 config_template + monitor_url 写入临时 YAML
  │
  ├─ 3. MASLauncher.launch()    ← 拉起 akg_rl_entry.py 子进程
  │     │
  │     ▼  (子进程内部)
  │     akg_rl_entry.py
  │       ├─ _inject_agent_env_vars()   ← 设置 AKG_AGENTS_* 环境变量
  │       ├─ load_config(dsl, backend)  ← 加载 akg_agents 配置
  │       └─ LangGraphTask.run()        ← 运行多智能体流程
  │            ├─ designer agent → POST /v1/chat/completions (model=kernel_designer)
  │            └─ coder agent   → POST /v1/chat/completions (model=kernel_gen)
  │
  │     ModelMonitor 拦截所有 LLM 请求：
  │       ├─ 按 model 字段识别 agent 角色
  │       ├─ 转发至 SlimeSglangBackend → slime SGLang router /generate
  │       ├─ 记录 InteractionRecord（含 token_ids, logprobs）
  │       └─ 返回 OpenAI 格式响应给子进程
  │
  ├─ 4. TrajectoryCollector.build()  ← 从 buffer 构造 EpisodeTrajectory
  │
  ├─ 5. RewardWorker.compute()       ← 调用 AKGKernelRewardProvider
  │
  └─ 6. 返回 EpisodeResult
         │
         ▼
generate_akg_episode() 后处理：
  ├─ 提取 kernel_gen 最后一轮的 response
  ├─ 截断至 max_response_tokens（防止 OOM）
  ├─ 填充 sample.tokens / response / response_length / rollout_log_probs / reward
  └─ 返回 slime Sample → 进入 GRPO 训练
```

### 2.2 异步并发模型

slime 的 rollout 循环通过 `asyncio` 并发调用 `generate_akg_episode()`。每个 sample 独立拥有：

| 资源 | 说明 |
|------|------|
| `AgentPipe` 实例 | 每次调用新建 |
| `ModelMonitor` | 监听独立的临时端口（port=0 由 OS 分配） |
| MAS 子进程 | 独立的 `akg_rl_entry.py` 进程 |
| episode_id | UUID，贯穿整条链路 |

共享资源（进程级单例，通过 `_PIPE_STATE` 缓存）：

| 资源 | 说明 |
|------|------|
| `AgentPipeConfig` | 命令模板、config 模板、model_mapping |
| `SlimeSglangBackend` | 指向 slime SGLang router 的推理后端 |
| `tokenizer` | HuggingFace tokenizer |
| `AKGKernelRewardProvider` | reward 计算器 |

这种设计保证多个 episode 可以并行执行，互不干扰。

## 3. 核心模块

### 3.1 `slime_generate.py` — slime 接入层

**职责：** 作为 slime `--custom-generate-function-path` 的入口函数。

**核心函数：** `generate_akg_episode(args, sample, sampling_params, evaluation=False) -> Sample`

**执行流程：**

1. 首次调用时通过 `_ensure_pipe_state(args)` 初始化共享状态：
   - 从 `args.hf_checkpoint` 加载 tokenizer
   - 创建 `SlimeSglangBackend`（指向 `args.sglang_router_ip:sglang_router_port`）
   - 加载 `akg_config_template.yaml` 作为 config 模板
   - 构建 `AgentPipeConfig`（命令模板指向 `akg_rl_entry.py`）
2. 从 `sample.prompt` 解析 KernelBench 任务（`op_name` + `task_desc`）
3. 调用 `AgentPipe.run()` 执行完整 episode
4. 从 trajectory 中提取 `kernel_gen` 最后一轮的 response
5. 截断 response token 至 `max_response_tokens`（防止 GPU OOM）
6. 填充 `sample.tokens`、`sample.response`、`sample.rollout_log_probs`、`sample.reward` 等字段
7. 如果 AgentPipe 失败，降级到 `_fallback_generate()`（直接用 slime 默认生成 + 启发式 reward）

**容错机制：** 任何 AgentPipe 异常都会被捕获并降级到 fallback，保证训练循环永不中断。

### 3.2 `akg_rl_entry.py` — MAS 子进程入口

**职责：** 由 `MASLauncher` 作为子进程拉起，运行真实的 `akg_agents` 多智能体系统。

**关键设计：**

- **始终返回 exit code 0**：负样本需要进入 reward 计算，非零退出会导致 episode 被丢弃。正确性由 `AKGKernelRewardProvider` 判断。
- **环境变量注入**：`_inject_agent_env_vars()` 将 ModelMonitor URL 注入到 `AKG_AGENTS_*` 环境变量中：
  - `AKG_AGENTS_STANDARD_*` → model_name=kernel_gen（coder 角色使用）
  - `AKG_AGENTS_COMPLEX_*` → model_name=kernel_designer（designer 角色使用）
  - 两组环境变量指向同一个 ModelMonitor URL，但使用不同的 model_name，使 ModelMonitor 能区分请求来源
- **agent_model_config 映射**：`designer → complex`，`coder → standard`，将 agent 角色映射到 `akg_agents` 的模型级别

### 3.3 `slime_sglang_backend.py` — 推理后端适配层

**职责：** 实现 `agent_trajectory_engine.InferenceBackend` 接口，将 OpenAI 风格的 chat completion 请求转换为 slime SGLang router 的原生 `/generate` API。

**转换过程：**

```
ModelMonitor 收到 OpenAI 请求
  │  messages: [{role, content}, ...]
  │  generation_params: {temperature, max_tokens, ...}
  ▼
SlimeSglangBackend.generate()
  │
  ├─ apply_chat_template(messages) → prompt_text
  ├─ tokenizer.encode(prompt_text) → input_ids
  ├─ 构建 SGLang payload: {input_ids, sampling_params, return_logprob: true}
  ├─ POST router_url/generate
  │
  ▼  解析响应
  ├─ text → response content
  ├─ meta_info.output_token_logprobs → [(logprob, token_id), ...]
  └─ 返回 ModelResponse(content, token_ids, logprobs, finish_reason)
```

**为什么需要这一层：** slime 的 SGLang router 使用原生 API（`input_ids` + `sampling_params`），不提供 OpenAI 兼容的 `/v1/chat/completions`。而 `akg_agents` 内部使用 OpenAI SDK 发请求。`SlimeSglangBackend` 在两者之间做协议转换。

### 3.4 `akg_kernel_reward.py` — 奖励计算

**职责：** 基于 `EpisodeTrajectory` 计算多目标奖励。

**奖励公式：**

```
reward = α · r_correct + β · r_perf · r_correct + γ · r_iter
```

| 分量 | 含义 | 当前状态 |
|------|------|---------|
| `r_correct` | Verifier 通过为 1，否则为 0 | 需要 `verifier_factory`，当前未配置时返回 0 |
| `r_perf` | 性能加速比 | Phase 2 启用，当前为 0 |
| `r_iter` | 迭代效率：`1 - n_turns / max_turns` | 已启用 |

**默认参数：** `α=1.0, β=0.0, γ=0.1, max_turns=5`

当前阶段由于 verifier 未配置，实际 reward = `γ · r_iter = 0.1 × (1 - 1/5) = 0.08`。

### 3.5 `trajectory_to_sample.py` — 轨迹到样本转换

**职责：** 将 `EpisodeResult` 转换为 slime `Sample` 所需的字段字典。

**Phase 1 策略（`last_turn`）：**

- 仅提取 `kernel_gen` 最后一个 turn
- `loss_mask = [1] * response_length`（只在 response 上计算 loss）
- `reward = episode.final_reward`

**后续可扩展策略：**

- `aggregate_role`：聚合 `kernel_gen` 的多轮输出，中间 turn 也参与训练
- `multi_role`：同一 episode 为 designer 和 coder 分别生成样本

> 注：当前 `slime_generate.py` 中内联实现了 `last_turn` 逻辑，`trajectory_to_sample.py` 作为独立模块提供更灵活的接口，供后续阶段使用。

## 4. `agent_trajectory_engine` 核心组件

### 4.1 组件关系

```
AgentPipe（编排器）
  ├── ModelMonitor（HTTP 代理 + 轨迹记录器）
  │     ├── 暴露 /v1/chat/completions
  │     ├── 按 model 字段路由到 model_mapping
  │     ├── 调用 InferenceBackend.generate()
  │     └── 记录 InteractionRecord 到 buffer
  ├── MASLauncher（子进程管理）
  │     ├── prepare_config() → 写临时 YAML
  │     ├── launch() → subprocess.Popen
  │     └── wait() → 等待退出
  ├── TrajectoryCollector
  │     └── build(buffer) → EpisodeTrajectory
  └── RewardWorker
        └── compute(trajectory, provider) → EpisodeResult
```

### 4.2 数据流

```
InteractionRecord (per LLM call)
  │  agent_role, turn_index, messages, response_text, token_ids, logprobs
  ▼
EpisodeTrajectory (per episode)
  │  episode_id, agent_trajectories: {role: [TurnData, ...]}
  ▼
EpisodeResult (per episode)
  │  trajectory, rewards, final_reward, status, metadata
  ▼
slime Sample (per training step)
     tokens, response, response_length, rollout_log_probs, reward, status, metadata
```

## 5. GPU 资源分配与训练配置

### 5.1 硬件配置

- 8× RTX 3090（24 GB 显存）
- 模型：Qwen3-4B-Instruct-2507

### 5.2 GPU 分配

| 角色 | GPU 数量 | 说明 |
|------|---------|------|
| 训练（Megatron-LM） | 4 (GPU 0-3) | TP=4, PP=1 |
| 推理（SGLang） | 4 (GPU 4-7) | 4 个独立 engine, 各 1 GPU |

### 5.3 关键训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `rollout-batch-size` | 2 | 每批 rollout 的 sample 数 |
| `n-samples-per-prompt` | 2 | 每个 prompt 生成的样本数 |
| `global-batch-size` | 4 | 训练 batch size |
| `num-rollout` | 10 | 总 rollout 轮数 |
| `rollout-max-response-len` | 1024 | 最大 response token 数 |
| `max-tokens-per-gpu` | 1536 | 每 GPU 最大 token 数（防 OOM） |
| `lr` | 1e-6 | 学习率 |
| `advantage-estimator` | grpo | GRPO 算法 |
| `eps-clip` | 0.2 | PPO clip 范围 |

### 5.4 检查点

| 类型 | 路径 |
|------|------|
| HuggingFace 原始 | `/data1/models/Qwen/Qwen3-4B-Instruct-2507` |
| torch_dist（Megatron） | `/data1/lmy/agentic-rl/checkpoints/Qwen3-4B-Instruct-2507-torch-dist` |
| 训练输出 | `/data1/lmy/agentic-rl/checkpoints/Qwen3-4B-Instruct-2507-slime` |

## 6. 目录结构

```
AscendAgentRL/
├── ascend_agent_rl/
│   └── agent_trajectory_engine/     # 轨迹引擎（共享组件）
│       ├── __init__.py
│       ├── pipe.py                  # AgentPipe + AgentPipeConfig
│       ├── gateway.py               # ModelMonitor
│       ├── backend.py               # InferenceBackend 抽象类 + VLLMBackend
│       ├── launcher.py              # MASLauncher
│       ├── collector.py             # TrajectoryCollector
│       ├── datatypes.py             # EpisodeResult, EpisodeTrajectory, TurnData, ...
│       ├── reward.py                # RewardProvider, RewardWorker
│       └── replay_cache.py          # ReplayCache
├── examples/
│   └── akg_kernel_gen/
│       ├── run_qwen3_4b_akg.sh      # 启动脚本
│       ├── README.md
│       ├── configs/
│       │   └── akg_config_template.yaml   # MAS 配置模板
│       ├── mas_entry/
│       │   ├── __init__.py
│       │   └── akg_rl_entry.py            # MAS 子进程入口（LangGraphTask）
│       └── ascendrl_glue/
│           ├── __init__.py
│           ├── slime_generate.py          # slime custom_generate 入口
│           ├── slime_sglang_backend.py    # SGLang 原生 API 适配
│           ├── akg_kernel_reward.py       # 多目标 reward 计算
│           ├── trajectory_to_sample.py    # EpisodeResult → Sample 转换
│           └── kernelbench_jsonl.py       # KernelBench 数据集转 JSONL
├── data/
│   └── kernelbench_level1.jsonl           # 训练数据
├── third_party/
│   └── slime/                             # slime v0.2.2（Git submodule）
├── doc/
│   ├── akg_agent_design.md                # 原始设计文档
│   └── akg_agent_implementation.md        # 本文档
└── pyproject.toml
```

## 7. 与 OrchRL 的差异对照

### 7.1 保持不变

| 组件 | 说明 |
|------|------|
| `akg_rl_entry.py` 核心职责 | subprocess 启动 LangGraphTask，始终 exit 0 |
| KernelBench 任务格式 | `{"op_name": "...", "task_desc": "..."}` |
| Reward 计算语义 | `α·r_correct + β·r_perf·r_correct + γ·r_iter` |
| `agent_trajectory_engine` | episode 采集逻辑不变 |
| black-box MAS 执行模式 | `akg_agents` 作为外部子进程运行 |

### 7.2 替换的部分

| OrchRL | AscendAgentRL |
|--------|--------------|
| `KernelBenchLoader` 提供 step batch | slime JSONL prompt source |
| `MateRolloutAdapter` 收集 episode | `slime_generate.py` 内 `AgentPipe` |
| VeRL trainer 消费 rollout 结果 | slime GRPO trainer |
| `VLLMBackend` → OpenAI `/v1/chat/completions` | `SlimeSglangBackend` → SGLang `/generate` |
| VeRL 管理推理服务 | slime 管理 SGLang router + engines |

### 7.3 新增的部分

| 模块 | 说明 |
|------|------|
| `SlimeSglangBackend` | OpenAI 消息 → SGLang 原生 API 协议转换 |
| `slime_generate.py` | slime `custom_generate` 入口，含 fallback 机制 |
| `run_qwen3_4b_akg.sh` | GPU 环境下的 slime + Ray 启动脚本 |
| WandB 集成 | 通过 slime 原生 `--use-wandb` 支持 |

## 8. 关键设计决策

### 8.1 为什么保留 `agent_trajectory_engine` 而不直接用 slime router

slime 的 router/gateway 解决的是**模型请求转发与训练期 metadata 保留**。而 `agent_trajectory_engine` 解决的是**跨角色、跨轮次、跨进程的 episode 级轨迹采集与结构化**。两者不是重复关系，而是分层互补：

- slime router 只能看到"某一次模型请求"
- `agent_trajectory_engine` 表达的是"这一次请求在整个 episode 里的角色、轮次、上下文位置，以及它和最终 reward 的关系"

### 8.2 为什么 MAS 子进程始终 exit 0

GRPO 需要同时看到正样本和负样本来估计 advantage。如果子进程非零退出导致 episode 被丢弃，训练只能看到成功的案例，无法学习"什么是错误的"。正确性判断交由 `AKGKernelRewardProvider`。

### 8.3 为什么使用 per-level 环境变量而非 per-agent

`akg_agents` 的配置系统基于**模型级别**（complex/standard/fast），不支持 per-agent 的环境变量。通过将 designer 映射到 `complex` 级别、coder 映射到 `standard` 级别，并为每个级别设置不同的 `MODEL_NAME`，实现了 ModelMonitor 对请求来源的区分。

### 8.4 为什么需要 response 截断

`akg_agents` 的多智能体对话可能产生非常长的响应（实测可达 16000+ token）。在 GRPO 训练中，entropy 计算需要 `O(seq_len × vocab_size)` 的显存，长序列会导致 GPU OOM。因此在 `slime_generate.py` 中将 response 截断到 `max_response_tokens`（默认 1024）。

### 8.5 fallback 机制的设计考量

当 AgentPipe 执行失败时（例如 `akg_agents` 依赖缺失、子进程超时等），`_fallback_generate()` 会调用 slime 默认的 SGLang 生成，加上启发式 reward。这保证了：

- 训练循环永不中断
- 即使 agent 管线有问题，模型仍然在学习基本的代码生成能力
- 通过 `sample.metadata["reward_type"]` 可以在 WandB 中区分两种来源

## 9. 训练阶段规划

### Phase 1：单角色训练闭环（已完成）

- 数据：KernelBench Level 1（100 tasks）
- 训练角色：`kernel_gen`（代码生成）
- 冻结角色：`kernel_designer`（设计规划）
- Sample 策略：`last_turn`
- Reward：`r_iter`（verifier 待接入后加入 `r_correct`）
- 目标：建立端到端可运行的训练闭环

### Phase 2：多轮训练增强

- 启用 `aggregate_role` 策略
- `kernel_gen` 的多轮输出都参与训练
- 更精细的 `loss_mask`（只在 assistant token 上计算 loss）
- 启用 `r_correct`（接入 KernelVerifier）
- 可选启用 `r_perf`（性能 profiling reward）

### Phase 3：双角色联合训练

- 同时训练 `kernel_designer` 和 `kernel_gen`
- 一个 episode 产出两个 sample
- 需要 `custom_convert_samples_to_train_data` 处理 group normalization
- 支持不同 tokenizer 的 role-level 路由

## 10. 监控与可观测性

### WandB 指标

训练过程通过 WandB 实时监控，关键指标：

| 指标 | 含义 |
|------|------|
| `rollout/rewards` | 归一化后的 reward |
| `rollout/raw_reward` | 原始 reward（来自 AKGKernelRewardProvider） |
| `rollout/response_lengths` | 平均 response 长度 |
| `rollout/truncated` | 被截断的比例 |
| `perf/rollout_time` | rollout 耗时 |
| `perf/actor_train_time` | 训练步耗时 |
| `perf/step_time` | 总步耗时 |

### Episode 级日志

每个 episode 完成后输出结构化日志：

```
[slime_generate] episode=<id> op=<op_name> turns=<n> reward=<r> status=<s> response_len=<l>
```

通过 `sample.metadata` 保留的信息支持事后分析：

- `episode_id`：唯一标识，可追溯到 trajectory
- `reward_type`：`akg_kernel_reward` 或 `fallback_heuristic`
- `turn_count`：episode 内 kernel_gen 的交互轮数
- `op_name`：KernelBench 任务名

## 11. 启动方式

```bash
# 设置 WandB API Key
export WANDB_API_KEY="your-key"

# 从 AscendAgentRL 根目录启动
cd /data1/lmy/agentic-rl/AscendAgentRL
bash examples/akg_kernel_gen/run_qwen3_4b_akg.sh
```

脚本会自动完成：清理残留进程 → 启动 Ray 集群 → 提交训练 job → 初始化 SGLang 推理引擎 → 开始 rollout + 训练循环。
