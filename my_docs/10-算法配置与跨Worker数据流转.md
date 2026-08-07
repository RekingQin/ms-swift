# 10 · 算法配置与跨 Worker 数据流转（RLHF 配置 / rollout / 分布式流转）

> 返回索引：[README.md](README.md)
> 前置阅读：[03-优化算法.md](03-优化算法.md)
> 本文回答：GRPO 等 RL 算法有哪些可调参数？数据如何从 trainer 流向 vLLM 引擎再流回？多机/多 worker 怎么流转？

---

## 1. RLHF 配置体系

### 1.1 配置类层次（`swift/rlhf_trainers/arguments.py`）

```text
HfGRPOConfig (trl)
  ▲
GRPOArgumentsMixin (args_mixin.py L246)     # GRPO 专属 + rollout 参数
  ▲
RolloutTrainerArgumentsMixin (L100)         # vLLM/生成参数
  ▲
VllmArguments (L8)                          # vLLM 引擎参数
  ▲
TrainArgumentsMixin (trainers/arguments.py) # 训练超参
  ▲
GRPOConfig (arguments.py L97)
```

各 rlhf_type 的 Config：`DPOConfig`（L29）/ `CPOConfig`（L42）/ `ORPOConfig`（L50）/ `KTOConfig`（L58）/ `RewardConfig`（L66）/ `PPOConfig`（L74）/ `GKDConfig`（L82）/ `GRPOConfig`（L97），均继承 `TrainArgumentsMixin + HfXXXConfig`。

### 1.2 vLLM 引擎参数（`VllmArguments`，args_mixin.py L8）

| 参数 | 默认 | 说明 |
|------|------|------|
| `vllm_gpu_memory_utilization` | 0.9 | 显存利用率 |
| `vllm_tensor_parallel_size` | 1 | TP |
| `vllm_pipeline_parallel_size` | 1 | PP |
| `vllm_enable_expert_parallel` | False | EP（MoE） |
| `vllm_max_num_seqs` | 256 | 最大并发序列 |
| `vllm_max_model_len` | None | 最大模型长度 |
| `vllm_disable_custom_all_reduce` | True | 自定义 all-reduce |
| `vllm_enforce_eager` | False | 关闭 CUDA graph |
| `vllm_max_lora_rank` | 16 | LoRA 注入 |
| `vllm_enable_prefix_caching` | None | 前缀缓存 |
| `vllm_use_async_engine` | None | 异步引擎 |
| `vllm_quantization` | None | 量化 |
| `vllm_speculative_config` | None | 投机采样 |
| `vllm_data_parallel_size` | 1 | DP |

### 1.3 Rollout 参数（`RolloutTrainerArgumentsMixin` L100）

| 参数 | 默认 | 说明 |
|------|------|------|
| `use_vllm` | False | 是否用 vLLM |
| `vllm_mode` | None | `server` / `colocate` |
| `vllm_server_base_url` | None | server 模式地址 |
| `vllm_server_port` | [8000] | server 端口 |
| `enable_flattened_weight_sync` | True | 扁平化权重同步 |
| `async_generate` | False | 异步生成 |
| `sleep_level` | 0 | vLLM 引擎睡眠级别 |
| `offload_optimizer` / `offload_model` | False | 显存卸载 |
| `generation_batch_size` | None | 生成批大小 |
| `steps_per_generation` | None | 每次生成的训练步数 |
| `top_k` / `top_p` / `min_p` | -1/1.0/0.0 | 采样参数 |
| `repetition_penalty` | 1.0 | 重复惩罚 |
| `stop_words` | [] | 停止词 |

### 1.4 GRPO 算法参数（`GRPOArgumentsMixin` L246）

| 参数 | 默认 | 算法 | 说明 |
|------|------|------|------|
| `epsilon` / `epsilon_high` / `delta` | 0.2/None/None | PPO 系 | clip 范围（delta 为 DAPO 上界） |
| `cosine_min/max_len_value_*` | -0.5~1.0 | CosineReward | 长度余弦奖励拐点 |
| `repetition_n_grams` / `repetition_max_penalty` | 3/-1.0 | 重复惩罚 | |
| `reward_model` / `reward_model_plugin` | None | RM | 奖励模型列表 + 插件 |
| `chord_sft_dataset` 等 | [] | CHORD | CHORD SFT 数据与 μ 调度 |
| `sync_ref_model` / `ref_model_sync_steps` | False/512 | ref 同步 | 周期同步 ref 权重 |
| `multi_turn_scheduler` / `max_turns` | None | 多轮 | 多轮调度器 |
| `dynamic_sample` / `max_resample_times` / `overlong_filter` | False/3/False | DAPO | 动态采样/超长过滤 |
| `soft_max_length` / `soft_cache_length` | None | SoftOverlong | |
| `scale_rewards` | None | Dr.GRPO/GDPO | group/batch/none/gdpo |
| `top_entropy_quantile` | 1.0 | Entropy-Reg | entropy 过滤 |
| `importance_sampling_level` | 'token' | GSPO | token/sequence/sequence_token |
| `tau_pos` / `tau_neg` | 1.0/1.05 | SAPO | 软门控温度 |
| `advantage_estimator` | 'grpo' | RLOO/R++ | grpo/rloo/reinforce_plus_plus |
| `teacher_kl_coef` | 1.0 | OPD-RL | teacher 信号系数 |
| `advantage_reweight` / `rlsd_lambda` 等 | None/0.5 | RLSD | 重加权 |
| `sdar_loss_coef` / `sdar_gate_beta` | 0.0/5.0 | SDAR | 蒸馏损失 |
| `real_tau` | 0.5 | REAL | |
| `fipo_decay_rate` 等 | 32.0 | FIPO | Future-KL |
| `num_generations_eval` | None | 评估 | |
| `loss_type` | 'grpo' | 全部 | grpo/dapo/bnpo/dr_grpo/fipo/cispo/sapo/real |

### 1.5 GRPOConfig 的 `__post_init__` 特化（arguments.py L101-123）

```text
require_version('trl>=0.26')
跳过 trl 的 __post_init__（它硬性要求 num_generations>=2）
禁用 vllm_reasoning_parser
cosine_max_len 默认 = max_completion_length
deepspeed zero3 时 stage3_prefetch_bucket_size=0（issue #3237）
dataloader_drop_last=True（issue #3863）
```

---

## 2. 跨 Worker 数据流转

### 2.1 单机多卡（torchrun + accelerate）

```text
trainer (主进程, rank 0)
  ├─ dataloader 采样 → 每 rank 一份（DataLoaderShard / BatchSamplerShard）
  ├─ rollout: 各 rank 用本进程 vLLM 引擎生成（colocate）或调 server
  ├─ 奖励: 各 rank 本地算 → gather 聚合
  ├─ advantage: 需跨进程 gather rewards 后统一算（compute_advantages 输入已 gather）
  └─ 梯度: accelerator.backward → DDP/FSDP/DeepSpeed all-reduce
```

### 2.2 数据聚合工具

| 工具 | 位置 | 用途 |
|------|------|------|
| `gather` / `gather_object` | `accelerate.utils` | 跨 rank 聚合张量/对象 |
| `get_even_process_data` | `rlhf_trainers/utils.py` | 样本均分各进程 |
| `DataLoaderDispatcher` | `swift/dataloader/dispatcher.py` L8 | rank0 分发 batch 到各 rank（`dist.scatter_object_list`） |
| `gather_for_unpadded_tensors` | `trainers/utils.py` L300 | SP/DP 下 gather 非 padded 张量 |
| `MeanMetric` | `metrics/utils.py` L73 | 跨 rank 聚合指标 |

### 2.3 关键：advantage 的跨进程计算

`grpo_trainer.py` `_compute_advantages` 流程：

```text
各 rank 本地 rewards [local_N, n_funcs]
    ↓ gather 到全局
全局 rewards [N, n_funcs]
    ↓ compute_advantages (advantage.py L10)   # 需全局才正确（组内归一化）
全局 advantages [N]
    ↓ 按 rank 切回本地
本地 advantages → expand_advantage_to_per_token
```

### 2.4 vLLM 引擎数据流转（`swift/infer_engine/`）

```text
GRPOSample.to_infer_request (data.py L244)
    ↓ 生成 RolloutInferRequest（messages + images + uuid）
InferEngine.infer (infer_engine.py L176, asyncio 并发 _batch_infer_stream)
    ↓
VllmEngine._infer_stream_async / _infer_full_async
    ↓
RolloutOutput（choices + response_token_ids + logprobs + finish_reason）
    ↓
sample.apply_rollout_output (data.py L195)  # 合并回样本
```

### 2.5 两种 vLLM 部署模式

| 模式 | 说明 | 参数 |
|------|------|------|
| `colocate` | vLLM 与训练同进程，`GRPOVllmEngine` | `--vllm_mode colocate` |
| `server` | 独立 vLLM server（FastAPI），`SwiftRolloutDeploy` | `--vllm_mode server --vllm_server_base_url ...` |

colocate 模式要点：

- `_prepare_vllm_engine`（rollout_mixin L459）；
- LoRA 请求管理（`GRPOVllmEngine` L25）；
- 权重同步：`enable_flattened_weight_sync` 扁平化同步 actor 权重到 engine。

### 2.6 多轮调度（`swift/rollout/multi_turn.py`）

- `RolloutScheduler` / `MultiTurnScheduler`（L23）；
- `async_infer`（L66）：并发请求 vLLM；
- `run`（L138/223）：用 `GRPOVllmEngine._batch_infer_stream` 并发；
- agent 多轮：`run_multi_turn`（agent_loop.py L78，asyncio.new_event_loop）。

### 2.7 跨节点流转（Megatron / Ray）

```text
Megatron（swift/megatron/）:
  TP/PP/CP/EP 在节点内切分，数据按 DP 分发
  checkpoint 用 dist_checkpointing 分片（见 04 篇）

Ray（swift/ray/）:
  Megatron-Ray 支持 GRPO/GKD（README 2026.06.10）
  ray_utils 的 try_init_ray 在 CLI 入口初始化
  跨 worker 通过 Ray 对象存储 / ray.put / ray.get 流转
```

---

## 3. 跨 Worker 数据流转全链路图（GRPO 为例）

```text
trainer (每个 rank)
  │
  ├─ dataloader rows
  ├─ GRPOSample.from_row (data.py L151)
  ├─ to_infer_request (L244)
  ├─ ──────────────── 跨进程/引擎边界 ────────────────
  │        ↓ InferEngine（本进程 colocate 或远端 server）
  ├─ RolloutOutput
  ├─ apply_rollout_output (L195) → response_token_ids/logprobs/finish_reason
  ├─ score_completions (grpo_algorithm.py L90) → rewards_per_func [local]
  ├─ ──────────────── gather 到全局 ────────────────
  ├─ compute_advantages (advantage.py L10) → advantages [global]
  ├─ ──────────────── 切回本地 ────────────────
  ├─ expand_advantage_to_per_token (L253) → [B, T]
  ├─ collate_to_grpo_micro_batch → (model_inputs, GRPOBatch)
  ├─ compute_loss (grpo_trainer.py L862) → loss
  └─ accelerator.backward → 分布式 all-reduce
```

---

## 4. 配置调优速查

| 目标 | 参数 |
|------|------|
| 提速生成 | `vllm_use_async_engine`、`async_generate`、`vllm_max_num_seqs`、TP 对齐 |
| 省显存 | `vllm_gpu_memory_utilization`、`sleep_level`、`offload_model/optimizer`、`vllm_enforce_eager` |
| 稳定性 | `dataloader_drop_last`、zero3 `stage3_prefetch_bucket_size=0` |
| 长响应 | `max_completion_length`、`cosine_max_len`、`stop_words` |
| 算法切换 | `loss_type`（grpo/dapo/...）、`advantage_estimator`、`scale_rewards` |
| 多轮 | `multi_turn_scheduler`、`max_turns`、`completion_length_limit_scope` |
| ref 同步 | `sync_ref_model`、`ref_model_sync_steps` |

---

## 5. 一句话总结

```text
配置 → dataclass 多重继承（VllmArguments → RolloutTrainerArgumentsMixin → GRPOArgumentsMixin → GRPOConfig）
流转 → 样本(GRPOSample) → 引擎请求(RolloutInferRequest) → 输出(RolloutOutput) → 奖励 → gather → advantage → loss
模式 → colocate（同进程）/ server（独立 FastAPI）/ 多轮调度 / Megatron-Ray
```

深入阅读：
- [03-优化算法.md](03-优化算法.md)：算法数学细节。
- [11-预训练、SFT和RL实现详解.md](11-预训练、SFT和RL实现详解.md)：GRPO 训练循环完整实现。
- [12-流式数据处理.md](12-流式数据处理.md)：数据如何被构造为样本。
