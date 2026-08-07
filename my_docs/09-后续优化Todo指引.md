# 09 · 后续优化 Todo 指引（结合业界进展）

> 返回索引：[README.md](README.md)
> 适用对象：计划深入 ms-swift 底层做优化与架构改进的开发者。
> 说明：本文给出结合代码现状与业界（vLLM/SGLang/verl/TRL/DeepSeek 等）进展的优化方向，按优先级排序。

---

## 1. 训练内核与调度

### 1.1 GRPO 训练性能（当前已很强，仍有空间）

现状（`swift/rlhf_trainers/grpo_trainer.py`）：

- 已实现：vLLM/SGLang/LMDeploy 异步引擎、liger 融合 loss、sequence_parallel 下的 RepeatSampler、`_compute_loss_chunked` 分块降显存（L1276）、off-policy IS correction。

优化方向：

- [ ] **generation batch 与 train batch 流水**：当前 `_prepare_inputs`（L189）一次性生成再拆分，可进一步 overlap rollout 与梯度计算（流水线），隐藏生成延迟。
- [ ] **多步生成缓存复用**：`steps_per_generation * num_iterations` 之间复用 completions，可调参减少生成次数，同时用 IS correction 控制偏差（已有 `rollout_importance_sampling_mode`）。
- [ ] **chunked loss 自动调优**：`_compute_loss_chunked` 的 chunk_size 目前取 per_device_train_batch_size，可改为按显存余量自适应。

### 1.2 序列并行结合

现状：Ulysses 已可与 ring-attention 结合（README），`get_packed_seq_params`（transformers_utils.py L344）供 padding-free + ring-attn。

优化方向：

- [ ] **Ring-Attention 的 kernel 层优化**：当前依赖 flash_attn varlen 接口，可对比 triton 自研 kernel 减少 lse 合并开销。
- [ ] **序列并行 + 梯度 checkpoint 的联合策略**：长序列场景下组合 activation offload 与 SP 的显存收益量化。
- [ ] **SP 下 MoE EP 的组合**：Megatron 已有 EP，transformers 后端的 SP+EP 可补。

---

## 2. rollout 引擎（infer_engine）

现状（`swift/infer_engine/`）：VllmEngine / GRPOVllmEngine / SglangEngine / LmdeployEngine / TransformersEngine。

优化方向：

- [ ] **权重同步去冗余**：GRPO 训练中 actor 权重 → vLLM engine 的同步，可借鉴 verl 的 delta checkpoint / 稀疏广播（字节级 diff）减少带宽。
- [ ] **vLLM sleep/wake 机制**：训练与 rollout 交替时，可让 vLLM 引擎进入低显存态（参考 verl 的显存卸载 + sleep/wake），支持更大模型 colocate。
- [ ] **多引擎负载均衡**：多 DP worker rollout server（`SwiftRolloutDeploy`）目前按 DP 分发，可加按负载的动态路由。
- [ ] **rolling batch / continuous batching 参数暴露**：把 vLLM 的 `max_num_batched_tokens`、`max_num_seqs` 等开放为可调参数并给默认建议。

---

## 3. 数据与训练效率

### 3.1 流式训练

现状（`swift/dataset/`）：`streaming` + `IterablePackingDataset`（L137，`packing_interval=128` 滑动打包）+ `DataLoaderDispatcher`（rank0 分发）。

优化方向：

- [ ] **流式 + 动态采样（DAPO）**：流式数据集下动态采样目前不兼容，可扩展 `resample.py` 支持 IterableDataset。
- [ ] **流式 shuffle 质量**：`shuffle_buffer_size` 默认值可自适应显存/内存。
- [ ] **多流 interleave**：`interleave_prob` 对 IterableDataset 的采样对齐。

### 3.2 多模态训练

现状：多模态 packing 提速 100%+，`MultimodalOptimizerCallback`（optimizers/multimodal.py L43）分 lr，freeze_vit/aligner/llm 可控。

优化方向：

- [ ] **视频/音频 token 压缩**：视频帧采样率、音频特征帧对齐策略可进一步优化显存。
- [ ] **多模态 packing 的 loss_scale 结合**：不同模态 token 的 loss 权重自动调节。

---

## 4. 显存优化

现状：Flash-Attn 2/3、liger、GaLore（`optimizers/galore/`）、gradient checkpointing（动态 `dynamic_gradient_checkpointing` L248）、activation cpu offload（`callbacks/activation_cpu_offload.py` L585）。

优化方向：

- [ ] **activation offload 精细化**：当前按"反向保存张量"整体 offload，可按层/按模块选择性 offload。
- [ ] **GaLore 在 MoE 上的适配**：GaLore 投影对 MoE 专家权重（大而稀疏）的收益评估。
- [ ] **ZeRO-offload 集成**：DeepSpeed 的 optimizer offload 与 GaLore 结合，进一步压显存。

---

## 5. 容错与可观测

现状：Flash Checkpoint（dlrover 异步）、GracefulExitCallback、`logging.jsonl`、PerfMetricsLogCallback（MFU/TFLOPS）。

优化方向：

- [ ] **checkpoint 校验**：保存后异步校验 `model.safetensors` 完整性（sum/大小），防止静默损坏。
- [ ] **resume 自动化**：`get_resume_checkpoint`（mixin.py L521）目前基于 `dlrover_latest.txt`，可扩展到失败重试自动恢复（exit code 非 0 时自动 resume）。
- [ ] **profiling 可视化**：`memory_time_profiling_context`（rlhf_trainers/utils.py L391）的耗时数据落盘为 chrome trace，可直接用 perfetto 查看。

---

## 6. 算法层

现状：GRPO 族（DAPO/GSPO/SAPO/CISPO/CHORD/RLOO/Reinforce++/FIPO/REAL/RLSD/SDAR/OPD-RL）已非常丰富。

优化方向（跟随业界）：

- [ ] **Dr.GRPO 的 token 级 KL 自适应**：`dr_grpo` loss_type 已有，可补动态 beta 调度。
- [ ] **多轮 RL 的 reward 稀疏处理**：`MultiTurnScheduler`（rollout/multi_turn.py L23）下长轨迹 reward shaping。
- [ ] **离线 RL（RLVR 数据）**：ms-swift 目前偏在线 on-policy，可扩展离线偏好数据 + PPO-offline。
- [ ] **scaling law 自动化**：基于 `PerfMetricsLogCallback` 的 MFU 数据，自动推荐 batch/SP 配置。

---

## 7. 工程化与生态

优化方向：

- [ ] **参数校验中心化**：目前 `rlhf_args.py` 中散落大量校验，可抽为统一的 `validate_args` 模块，报错含参数名+建议值。
- [ ] **CLI 配置模板**：`parse_yaml_args` 已支持 yaml，可加"配置模板合并"（base + override）。
- [ ] **模型注册自动发现**：`MODEL_MAPPING` 目前是显式注册，可加 `entry_points` 自动发现第三方模型。
- [ ] **测试覆盖率**：`tests/` 用 unittest，可逐步迁移 pytest + 增加 GRPO loss 数值单测（与 verl 结果对齐）。

---

## 8. 优先级路线图（建议）

```text
P0（收益最大、见效快）:
  ├─ GRPO 生成/训练流水 overlap
  ├─ actor→vLLM 权重同步去冗余（delta）
  └─ resume 自动化 + checkpoint 校验
P1（架构级）:
  ├─ vLLM sleep/wake + 显存卸载
  ├─ 流式 + 动态采样
  └─ activation offload 精细化
P2（探索性）:
  ├─ 离线 RL / 多轮 reward shaping
  ├─ 参数校验中心化 / CLI 模板
  └─ kernel 层（triton ring-attn 对比）
```

---

## 9. 一句话总结

```text
性能 → 流水 overlap + delta 权重同步 + SP/EP 组合
显存 → activation offload 精细化 + GaLore×MoE + vLLM sleep/wake
工程 → resume 自动化 + checkpoint 校验 + 参数校验中心化
算法 → 离线 RL + 多轮 reward shaping + 动态 beta
```

对照参考：
- verl 的 `checkpoint_engine/delta_checkpoint_engine.py`（字节级 diff + 稀疏广播）可作为权重同步参考。
- verl 的 `utils/memory_utils.py` / `activation_offload.py` 可作为显存优化参考。
