# AscendAgentRL AKG Kernel Agent Example 设计方案

## 1. 背景与目标

本文档用于设计 `AscendAgentRL` 中的 AKG Kernel Agent example，目标是在尽量复用现有组件的前提下，将 `OrchRL/examples/akg_kernel_gen` 的样例迁移为一个基于 `slime` 的可训练 recipe。

目标约束如下：

- `AscendAgentRL` 与 `OrchRL` 都是 agentic-rl 框架。
- 两者共同点是都基于 `agent_trajectory_engine` 无侵入采集 agent 与 LLM 的交互轨迹。
- 两者核心差异是训练框架不同：
  - `OrchRL` 基于 `VeRL`
  - `AscendAgentRL` 期望基于 `slime`
- 需要参考 `OrchRL/examples/akg_kernel_gen`，设计 `AscendAgentRL` 中等价的 AKG kernel agent example。
- 需要兼容 `/data1/lmy/agentic-rl/akg/akg_agents` 中现有多 agent 执行方式。

本设计优先追求：

- 最小迁移面
- 与现有 AKG black-box 执行方式兼容
- 保留无侵入轨迹采集
- 先支持单角色训练，再逐步扩展到双角色训练

## 2. 参考实现与结论

本方案主要参考以下实现：

- `OrchRL/examples/akg_kernel_gen/README.md`
- `OrchRL/examples/akg_kernel_gen/mas_entry/akg_rl_entry.py`
- `OrchRL/examples/akg_kernel_gen/orchrl_glue/akg_kernel_reward.py`
- `OrchRL/examples/akg_kernel_gen/orchrl_glue/kernelbench_loader.py`
- `AscendAgentRL/ascend_agent_rl/agent_trajectory_engine/*`
- `slime/examples/multi_agent/*`
- `slime/examples/tau-bench/*`
- `slime/docs/zh/get_started/customization.md`

分析后的核心结论如下：

1. `AscendAgentRL` 中已经有一套与 `OrchRL` 基本同构的 `agent_trajectory_engine`，因此这次适配的重点不是重写轨迹引擎，而是把 rollout 与训练 glue 从 VeRL 风格切到 slime 风格。
2. `slime` 的最佳接入点不是修改 trainer 主干，而是通过自定义 rollout/generate 接口接入复杂 agent episode。
3. 对 AKG 任务，最稳妥的第一阶段方案是：
   - 保持 AKG 作为 black-box subprocess
   - 继续使用 `agent_trajectory_engine` 捕获 episode trajectory
   - 在 slime rollout 侧把一个完整的 AKG episode 转换为一个可训练 `Sample`
4. 第一阶段应只训练 `kernel_gen`，冻结 `kernel_designer`，避免多角色 credit assignment、tokenizer 差异和 train-data reshape 带来的复杂度。

## 3. 设计原则

### 3.1 保持 AKG 执行形态不变

AKG 的多 agent 逻辑、Verifier、WorkerManager、LangGraphTask 都尽量不改，继续通过 subprocess 形式调用。这样可以最大化复用：

- `akg_agents` 的任务构建逻辑
- `KernelVerifier` 的 correctness 检查逻辑
- 已有 worker 运行环境

### 3.2 轨迹采集与训练解耦

沿用已有分层：

- episode 执行层：AKG subprocess
- 轨迹采集层：`agent_trajectory_engine`
- reward 层：基于 `EpisodeTrajectory`
- 训练层：slime

这样 `AscendAgentRL` 的 example 可以保持与 `OrchRL` 样例一致的语义，同时只替换训练 glue。

### 3.3 先做单角色可训练闭环

第一阶段只让 `kernel_gen` 产出训练样本，理由如下：

- slime 默认训练数据结构天然更适合“一条 response 对应一个 sample”
- `kernel_designer` 和 `kernel_gen` 后续可能使用不同 tokenizer 或不同服务路由
- 双角色同时训练需要明确 episode 内 reward 分配与 sample grouping 规则，复杂度较高

## 4. 推荐总体方案

### 4.1 推荐接入方式

推荐在 slime 中使用 `--custom-generate-function-path`，而不是一开始就完全重写 `--rollout-function-path`。

原因：

- slime 已经提供标准的 rollout 主流程
- `custom_generate` 允许把复杂的 agent episode 封装为一次“特殊生成”
- 接口更轻，侵入性更小
- 后续如果需要全自定义 rollout，再升级到 `rollout-function-path`

### 4.2 高层链路

推荐链路如下：

1. slime 从 KernelBench JSONL 中取出一条任务
2. `custom_generate` 读取样本中的 `op_name/task_desc`
3. `custom_generate` 调用 `AgentPipe.run(...)`
4. `AgentPipe` 启动 `ModelMonitor + MASLauncher + akg_rl_entry.py`
5. AKG subprocess 内部运行多 agent 流程
6. `ModelMonitor` 捕获各角色 LLM 请求与响应
7. `TrajectoryCollector` 构造 `EpisodeTrajectory`
8. `AKGKernelRewardProvider` 基于 trajectory 计算 reward
9. 将 `EpisodeResult` 转成 slime `Sample`
10. slime 对该 sample 执行后续 reward normalization、GRPO/PPO 训练

## 5. 与 OrchRL 方案的差异

### 5.1 保持不变的部分

- `mas_entry/akg_rl_entry.py` 的核心职责
- KernelBench 任务格式
- `AKGKernelRewardProvider` 的 reward 计算方式
- `agent_trajectory_engine` 的 episode 采集逻辑
- AKG black-box subprocess 执行模型

### 5.2 替换的部分

`OrchRL` 中：

- `KernelBenchLoader` 提供 step batch
- `MateRolloutAdapter` 负责收集 episode
- VeRL trainer 消费 rollout 结果

`AscendAgentRL` 中：

- slime 的 prompt/data source 提供输入样本
- `custom_generate` 执行 AKG episode
- slime 将 `Sample` 转换为训练数据

### 5.3 结论

本次迁移的本质不是“把 AKG 重新实现一遍”，而是把：

- `dataset -> episode -> trajectory -> reward -> train sample`

这条链从 `OrchRL + VeRL` 形式改写为：

- `dataset -> custom_generate -> AgentPipe -> EpisodeResult -> slime Sample`

## 5.4 关于"slime 已有 gateway/router，Trajectory Engine 是否冗余"的结论

这是本方案里的一个关键判断点。结合 `slime` 源码，结论是：

- `slime` 确实已经提供了 router/gateway 设计，用于把请求路由到一个或多个 SGLang worker
- 但 `slime` 的 router/gateway 解决的是“模型请求转发与训练期 metadata 保留”
- `agent_trajectory_engine` 解决的是“跨角色、跨轮次、跨进程的 episode 级轨迹采集与结构化”
- 对 AKG 这种 black-box 多 agent subprocess 场景，两者不是重复关系，而是分层互补关系

### 5.4.1 slime 现有能力的真实边界

从 `slime` 源码可以确认：

- 默认 rollout 的核心生成路径是 `slime.rollout.sglang_rollout.generate()`，其本质是：
  - 将 `Sample.prompt` tokenize
  - 发送到 router 的 `/generate`
  - 从返回结果中回填 `sample.tokens`、`sample.response`、`sample.response_length`、`sample.rollout_log_probs`
  - 再根据 `meta_info.finish_reason` 设置 `sample.status`
- `custom_generate` 只是默认 rollout 中的一个可替换钩子，入口签名为：
  - `async def custom_generate(args, sample: Sample, sampling_params: dict) -> Sample`
- `slime` 的插件契约测试也明确要求，自定义生成函数最终仍然要返回标准 `Sample`，并正确填充至少这些字段：
  - `tokens`
  - `response`
  - `response_length`
  - `reward`
  - `status`
- `slime` 的 multi-agent example 虽然允许 `custom_generate` 返回 `list[Sample]`，但它返回的已经是“训练样本”，不是统一的 episode trajectory 对象
- `slime` 的 tau-bench example 也是先在自定义环境里执行完整 interaction trajectory，再由环境侧逻辑自行转换成 `Sample`

这说明 `slime` 提供的是：

- rollout orchestration
- router / gateway
- `Sample` 训练契约
- 若干自定义 hook

但它并没有提供一个通用的、框架内建的：

- 多角色消息事件模型
- episode 级统一轨迹对象
- 对外部 black-box subprocess 的无侵入 LLM 调用监听层
- 面向 reward provider 的标准 `EpisodeTrajectory` 抽象

### 5.4.2 slime router / gateway 能解决什么，不能解决什么

`slime` 文档和实现里确实有 router/gateway 设计：

- `slime router` 是一个 rollout/data generation 阶段的轻量 HTTP proxy
- 它负责 worker 注册、请求路由、健康检查，以及中间件扩展
- 它还能保留训练期需要的额外 metadata，比如：
  - token-level logprobs
  - loss-mask 对齐所需 token 信息
  - MoE 的 `routed_experts`

但这一层的职责仍然是“请求代理与生成期 metadata 保真”，不是“episode 轨迹建模”。

尤其对 AKG 任务，下面这些问题并不是 router/gateway 能直接解决的：

- 一个 episode 内有多个 role：`kernel_designer`、`kernel_gen`、可能还有 verifier / environment 反馈
- LLM 调用发生在 AKG subprocess 内部，而不是 slime 主 rollout 线程里直接发起
- reward 计算依赖的不是单次 `/generate` 响应，而是“整个 episode 的结构化轨迹”
- 后续要支持单角色训练、多轮聚合、双角色训练，必须先有稳定的 episode 语义边界

换句话说：

- router/gateway 只能看到“某一次模型请求”
- `agent_trajectory_engine` 需要表达的是“这一次请求在整个 episode 里的角色、轮次、上下文位置，以及它和最终 reward 的关系”

### 5.4.3 为什么在 AKG 场景里 `agent_trajectory_engine` 仍然必要

对于 AKG kernel 生成任务，保留 `agent_trajectory_engine` 的核心理由有四个：

1. 保持对现有 AKG black-box 执行方式的兼容

- AKG 当前是通过 subprocess 运行多 agent 逻辑
- 如果完全改为直接由 slime rollout 驱动多角色消息流，迁移面会显著扩大
- `agent_trajectory_engine` 可以继续以无侵入方式监听 subprocess 内的模型调用

2. 提供统一的 episode 级数据结构

- reward provider 的输入更适合是 `EpisodeTrajectory`，而不是散落的 `Sample`
- `last_turn` / `aggregate_role` / `multi_role` 这些 sample 策略，本质上都依赖先有稳定 trajectory

3. 解耦 reward 语义与训练框架

- correctness / profiling / iteration bonus 本质上都属于 episode reward
- 如果把这些逻辑直接塞进 slime 的 `custom_generate`，会让“轨迹采集、reward 计算、Sample 拼装”混在一起
- 保留 `EpisodeTrajectory -> RewardProvider -> Sample` 这条链，模块边界更稳定

4. 保留可观测性与调试能力

- 对多 agent 系统，训练失败很多时候不是模型本身问题，而是角色交互、解析、Verifier、环境反馈链路的问题
- 只有保留结构化 trajectory，后续才能稳定做：
  - 失败 episode 回放
  - role-level 统计
  - reward breakdown 分析
  - token/sample 对 episode 的可追溯映射

### 5.4.4 对总体方案的影响

基于以上分析，推荐的职责边界应进一步明确为：

- `slime router / gateway`
  - 负责把训练期模型请求路由到正确的推理后端
  - 保留 token/logprob/routed_experts 等训练所需 metadata
- `slime custom_generate`
  - 负责把“一次 AKG episode 执行”接入 slime rollout 主流程
  - 最终返回一个或多个符合 `Sample` 契约的训练样本
- `agent_trajectory_engine`
  - 负责在 AKG subprocess 内部无侵入采集多角色 episode 轨迹
  - 向 reward provider 和 sample converter 暴露稳定的 `EpisodeTrajectory`

因此，本方案不是在 `slime` 已有能力之外“重复造一个 gateway”，而是：

- 复用 `slime` 的 router/gateway 做模型访问层
- 保留 `agent_trajectory_engine` 做 episode 采集层
- 在两者之间通过 `InferenceBackend` 和 `trajectory_to_sample` 完成桥接

### 5.4.5 进一步的实现含义

这个判断会直接影响实现方式：

- 不建议把 AKG 多 agent 协调逻辑整体搬进 slime 的 rollout 主循环
- 也不建议让 reward provider 直接面向零散 `Sample` 做 episode 反推
- 推荐继续保持：
  - `AKG subprocess -> agent_trajectory_engine -> EpisodeTrajectory -> RewardProvider -> trajectory_to_sample -> slime Sample`

只有在未来 AKG agent 本身也愿意改造成 slime-native 的多 agent 环境，并且：

- 各角色调用都由 slime rollout 直接驱动
- 训练只需要 sample-level 语义
- 不再依赖 subprocess black-box 兼容性

这时才有理由考虑弱化甚至移除 `agent_trajectory_engine`。在当前阶段，这样做并不划算，也会显著增加迁移风险。

## 6. 建议目录结构

建议在 `AscendAgentRL` 下新增如下目录：

```text
AscendAgentRL/
  examples/
    akg_kernel_gen/
      README.md
      run_qwen3_4b_akg.sh
      scripts/
        prepare_kernelbench_data.py
      configs/
        akg_config_template.yaml
        slime_akg_kernel_gen.yaml
      mas_entry/
        __init__.py
        akg_rl_entry.py
      ascendrl_glue/
        __init__.py
        akg_kernel_reward.py
        kernelbench_jsonl.py
        slime_generate.py
        slime_sglang_backend.py
        trajectory_to_sample.py
      tests/
        test_akg_rl_entry.py
        test_akg_kernel_reward.py
        test_kernelbench_jsonl.py
        test_trajectory_to_sample.py
```

说明：

- `mas_entry/akg_rl_entry.py` 负责 subprocess 内 AKG episode 执行
- `slime_generate.py` 负责 slime 接口适配
- `slime_sglang_backend.py` 负责把 slime router 接到 `InferenceBackend`
- `trajectory_to_sample.py` 负责将 `EpisodeResult` 转成 slime `Sample`

## 7. 核心模块设计

### 7.1 `mas_entry/akg_rl_entry.py`

职责：

- 读取 `MASLauncher.prepare_config()` 写出的 YAML
- 注入 AKG 所需环境变量
- 解析任务 JSON
- 构建 AKG LangGraphTask
- 运行 AKG episode
- 始终返回 exit code 0

保留始终 `exit 0` 的原因：

- 负样本需要进入 reward 计算
- 非零退出会导致 episode 在上游被直接丢弃
- correctness 的判断应由 reward provider 决定，而不是由进程退出码决定

建议最小改动：

- 复用 OrchRL 版本结构
- 仅把默认 task 配置从 CUDA 改为 Ascend：
  - `backend=ascend`
  - `arch=ascend910b2`
  - `dsl=triton_ascend`

### 7.2 `ascendrl_glue/akg_kernel_reward.py`

职责：

- 输入 `EpisodeTrajectory`
- 提取 `kernel_gen` 最后一轮响应
- 抽取代码块
- 调用 verifier 计算 correctness
- 可选启用 profiling reward
- 输出：

```python
{
    "agent_rewards": {...},
    "final_reward": float,
}
```

建议 reward 公式保持与 OrchRL 一致：

```text
reward = alpha * r_correct + beta * r_perf * r_correct + gamma * r_iter
```

其中：

- `r_correct`: verifier 通过为 1，否则 0
- `r_perf`: Phase 2 再启用
- `r_iter`: `1 - n_turns / max_turns`

### 7.3 `ascendrl_glue/slime_sglang_backend.py`

职责：

- 实现 `agent_trajectory_engine.backend.InferenceBackend`
- 将来自 `ModelMonitor` 的 OpenAI-style chat request 转发到 slime rollout router
- 负责：
  - chat template 拼装
  - tokenizer 编码
  - 调用 slime router `/generate`
  - 抽取 text、token_ids、logprobs
  - 返回 `ModelResponse`

为什么需要单独实现这一层：

- `agent_trajectory_engine` 现有 `VLLMBackend` 面向标准 OpenAI `/v1/chat/completions`
- slime 推理主链更自然的入口是 `/generate`
- AKG example 需要与 slime rollout 路由、tokenizer、采样参数保持一致

### 7.4 `ascendrl_glue/slime_generate.py`

职责：

- 作为 slime `custom_generate` 函数入口
- 输入单条 slime `Sample`
- 构造 `AgentPipeConfig`
- 调用 `AgentPipe.run(...)`
- 将 `EpisodeResult` 转换成 slime `Sample`

建议函数签名：

```python
async def generate_akg_episode(args, sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample:
    ...
```

### 7.5 `ascendrl_glue/trajectory_to_sample.py`

职责：

- 把 `EpisodeResult` 映射成 slime `Sample`
- 决定哪些 token 参与 loss
- 决定一个 episode 产出几个 trainable sample

第一阶段推荐模式：`last_turn`

规则：

- 仅提取 `kernel_gen` 的最后一个 turn
- 用该 turn 的 `messages + response_text` 重建训练样本
- `reward = episode.final_reward`
- `loss_mask = [1] * response_length`

第二阶段可扩展模式：

- `aggregate_role`：将一个 role 的多轮 turn 聚合为一个 sample
- `multi_role`：同一 episode 为多个 role 分别产出 sample

## 8. 数据格式设计

### 8.1 输入数据格式

建议继续沿用 OrchRL 里的任务语义：

```json
{
  "op_name": "1_relu",
  "task_desc": "class Model(nn.Module): ..."
}
```

slime 训练时可以直接读取 JSONL，每行一条：

```json
{"prompt":"{\"op_name\":\"1_relu\",\"task_desc\":\"class Model...\"}"}
```

或者也可以拆成字段：

```json
{"prompt":"...", "op_name":"1_relu", "task_desc":"class Model..."}
```

推荐保留 `prompt` 为 JSON 字符串，这样与 OrchRL 的现有 `mas_entry` 兼容最好。

### 8.2 输出训练样本格式

第一阶段推荐一条 episode 只输出一个 slime `Sample`。

样本关键字段如下：

- `prompt`: 当前训练 prompt
- `tokens`: prompt + response 的完整 token 序列
- `response`: `kernel_gen` 最后一次输出文本
- `response_length`: response token 数
- `reward`: `EpisodeResult.final_reward`
- `loss_mask`: `[1] * response_length`
- `status`: completed/truncated/failed
- `metadata`:
  - `episode_id`
  - `op_name`
  - `agent_rewards`
  - `turn_count`
  - `exit_code`
  - `raw_reward_breakdown`

## 9. Sample 生成策略

### 9.1 Phase 1：`last_turn` 策略

推荐策略：

- 只训练 `kernel_gen`
- 只对 `kernel_gen` 最后一轮回答做监督

优点：

- 最稳定
- 最贴近当前 slime 单 sample 假设
- 避免多轮中间状态对 loss mask 的复杂处理
- reward 可直接绑定到最后一次代码生成结果

适用场景：

- 初始闭环验证
- correctness reward 训练
- 单策略 GRPO

### 9.2 Phase 2：`aggregate_role` 策略

后续可扩展为：

- 聚合 `kernel_gen` 的多轮输出
- 每轮 assistant token 置 `loss_mask=1`
- 来自 designer/verifier/环境的文本 token 置 `loss_mask=0`

这样可以让模型学习多轮修正过程，而不仅是最终代码结果。

### 9.3 Phase 3：`multi_role` 策略

如果后续要同时训练 `kernel_designer` 与 `kernel_gen`，建议：

- 一个 episode 拆成两个 sample
- 每个 role 各自形成一条可训练样本
- 共享同一个 episode-level reward，或增加 role-specific shaping

这一阶段建议配套实现 slime 的：

- `custom_convert_samples_to_train_data`

否则 group normalization 与 sample grouping 会变复杂。

## 10. slime 配置方案

### 10.1 推荐启动参数

推荐通过以下方式接入：

- `--custom-generate-function-path examples.akg_kernel_gen.ascendrl_glue.slime_generate.generate_akg_episode`
- `--custom-config-path examples/akg_kernel_gen/configs/slime_akg_kernel_gen.yaml`

### 10.2 自定义配置文件

建议增加：

`examples/akg_kernel_gen/configs/slime_akg_kernel_gen.yaml`

示例：

```yaml
akg_config_template_path: examples/akg_kernel_gen/configs/akg_config_template.yaml
akg_mas_entry: examples/akg_kernel_gen/mas_entry/akg_rl_entry.py
akg_target_role: kernel_gen
akg_sample_mode: last_turn
akg_reward_alpha: 1.0
akg_reward_beta: 0.3
akg_reward_gamma: 0.1
akg_enable_profiling: false
akg_designer_backend_url: http://127.0.0.1:9000
akg_designer_model_name: kernel_designer
akg_kernel_gen_route: train
akg_task_backend: ascend
akg_task_arch: ascend910b2
akg_task_dsl: triton_ascend
akg_max_iterations: 5
```

说明：

- slime 的 `custom-config-path` 会把 YAML 顶层字段直接挂到 `args` 上
- 因此建议 AKG 配置采用扁平键名，避免与 slime 内置参数冲突

### 10.3 AKG config template

建议继续保留 OrchRL 风格：

```yaml
agents:
  kernel_gen:
    model: kernel_gen
    llm:
      base_url: ""
      api_key: "dummy"
  kernel_designer:
    model: kernel_designer
    llm:
      base_url: ""
      api_key: "dummy"

task:
  framework: torch
  backend: ascend
  arch: ascend910b2
  dsl: triton_ascend
  max_iterations: 5
```

## 11. 训练阶段划分

### Phase 1：MVP

目标：

- 建立可运行的单角色训练闭环

范围：

- 数据：KernelBench Level 1
- backend：Ascend
- dsl：`triton_ascend`
- 训练角色：`kernel_gen`
- reward：correctness + iteration bonus
- sample mode：`last_turn`

产物：

- 自定义 generate 接口
- AKG reward provider
- trajectory 到 sample 的转换器
- JSONL 数据准备脚本
- 启动脚本

### Phase 2：多轮训练增强

目标：

- 让 `kernel_gen` 学习多轮修正轨迹

范围：

- 启用 `aggregate_role`
- 更精细的 `loss_mask`
- 可选启用 profiling reward

### Phase 3：双角色联合训练

目标：

- 同时训练 `kernel_designer` 与 `kernel_gen`

范围：

- 多 role sample 生成
- role-specific tokenizer/route 支持
- custom convert to train data
- role-level reward 分配

## 12. 测试设计

建议测试层次与 OrchRL 版本保持一致。

### 12.1 `test_akg_rl_entry.py`

验证：

- subprocess 逻辑始终返回 0
- task 异常时不抛出非零退出
- env var 注入正确
- task JSON 解析正确

### 12.2 `test_akg_kernel_reward.py`

验证：

- reward payload 结构正确
- correctness reward 正确
- 空 trajectory 返回 0 reward
- 多 role reward 分配正确
- markdown code block 抽取正确

### 12.3 `test_kernelbench_jsonl.py`

验证：

- KernelBench 目录能正确转成 JSONL
- `prompt` 字段格式符合 `mas_entry` 预期
- batch 循环与 shuffle 可复现

### 12.4 `test_trajectory_to_sample.py`

验证：

- `last_turn` 模式的 token/response/reward 映射正确
- `loss_mask` 长度与 `response_length` 一致
- `metadata` 中 episode 信息保留正确
- failure/truncated 场景映射正确

### 12.5 可选集成测试

后续可增加 mock 版集成测试：

- fake backend
- fake verifier
- fake AKG subprocess

用于验证：

- `slime_generate -> AgentPipe -> reward -> Sample`

整条链路不依赖 NPU 与真实模型服务即可运行。

## 13. 风险与关键问题

### 13.1 tokenizer 对齐风险

最大风险之一是 tokenizer/chat template 不一致。

如果 `trajectory_to_sample` 使用的 tokenizer 与 rollout 期间实际使用的 tokenizer 不同，会导致：

- `tokens` 不一致
- `response_length` 错误
- `loss_mask` 错位

因此建议：

- `slime_sglang_backend.py` 与 `trajectory_to_sample.py` 共用同一个 tokenizer 来源

### 13.2 多角色 tokenizer 差异

如果 `kernel_designer` 与 `kernel_gen` 后续使用不同 tokenizer，则：

- Phase 1 只训练 `kernel_gen` 是安全的
- Phase 3 的双角色训练必须显式分 role 处理

### 13.3 verifier 运行环境依赖

`AKGKernelRewardProvider` 会依赖：

- worker manager
- Ascend/NPU 运行环境
- verifier 配置

因此必须保留 `verifier_factory` 注入方式，便于：

- 单测 mock
- CI 环境脱离真实硬件

### 13.4 episode 到 sample 的 credit assignment

一个 AKG episode 本质上是多轮、多角色、多类型消息交织。

如果直接把整段轨迹无差别作为一个训练样本，容易造成：

- 中间规划 token 与最终代码 token 共享同一 reward
- role 间 credit 混淆
- 训练信号噪声过大

因此本方案推荐先使用 `last_turn`。

## 14. 最终建议

### 14.1 推荐落地路径

最推荐的落地方案如下：

1. 在 `AscendAgentRL/examples/akg_kernel_gen` 新增完整样例目录
2. 复用 `agent_trajectory_engine` 作为 episode 执行与轨迹采集层
3. 新增 slime `custom_generate` 适配层
4. 第一阶段只训练 `kernel_gen`
5. 使用 `last_turn` 将 episode 转换为一个 slime `Sample`
6. reward 保持 OrchRL 版本语义，只把 verifier 运行环境切到 Ascend

### 14.2 为什么这是当前最优解

该方案同时满足：

- 最大化复用 `OrchRL` 现有 AKG 方案
- 最大化复用 `AscendAgentRL` 现有轨迹引擎
- 最小化对 slime 主框架的侵入
- 最快建立端到端可训练闭环
- 为后续多轮、多角色联合训练保留清晰扩展面

## 15. 后续实现建议

建议按以下顺序推进：

1. 先实现 `prepare_kernelbench_data.py`
2. 实现 `akg_rl_entry.py`
3. 实现 `akg_kernel_reward.py`
4. 实现 `slime_sglang_backend.py`
5. 实现 `slime_generate.py`
6. 实现 `trajectory_to_sample.py`
7. 先补齐单测
8. 最后补启动脚本与 README

如果需要继续推进实现，下一步建议直接输出“文件级开发任务拆解”，把每个文件的函数签名、输入输出和依赖关系展开成实现清单。
