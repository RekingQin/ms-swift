# 11 · 预训练、SFT 和 RL 实现详解（三大训练任务深入）

> 返回索引：[README.md](README.md)
> 前置阅读：[01-核心运行流程.md](01-核心运行流程.md)、[03-优化算法.md](03-优化算法.md)
> 本文回答：PT / SFT / RLHF 三大训练任务从数据到 loss 到保存的完整实现细节？

---

## 1. 三大任务的 Pipeline 差异

| 维度 | PT | SFT | RLHF |
|------|----|-----|------|
| Pipeline | `SwiftPretrain`（train/pretrain.py L11，继承 SwiftSft） | `SwiftSft`（train/sft.py L22） | `SwiftRLHF`（train/rlhf.py L23，继承 SwiftSft） |
| 参数类 | `PretrainArguments` | `SftArguments` | `RLHFArguments` |
| Trainer | `Seq2SeqTrainer`（task_type=causal_lm） | 同左 | `rlhf_type` 决定 |
| 模型 | 单模型 | 单模型 | 多模型（ref/reward/value/teacher） |
| 数据 | 支持 `truncation_strategy='split'` | Lazy/Packing/Encode | KTO 特殊预处理 / GRPO 延迟编码 |

---

## 2. SFT 实现详解

### 2.1 调用链

```text
swift sft --model Qwen3-8B --dataset alpaca-zh --lora_rank 8
    ↓
SwiftSft.__init__ (sft.py L26)
  ├─ _prepare_model_tokenizer (L48): args.get_model_processor()
  ├─ _prepare_template (L64): args.get_template() + set_mode('train')
  └─ _prepare_flash_ckpt
    ↓
SwiftSft.run (L159)
  ├─ _prepare_dataset (L96)
  ├─ prepare_model (TunerMixin L340): LoRA 注入
  ├─ TrainerFactory.get_trainer_cls → Seq2SeqTrainer
  └─ train(trainer) (L254) → trainer.train(resume_checkpoint)
```

### 2.2 数据准备（`_prepare_dataset` L96）

```text
_get_dataset:
  DatasetSyntax 解析 (dataset:subset#sample)
  DatasetLoader.load_dataset (loader.py L224)
  concat / interleave / shuffle
_encode_dataset:
  RowPreprocessor 系列标准化行
  template.encode 编码
_post_process_datasets (L125):
  LazyLLMDataset（惰性编码跳过坏样本）
  PackingDataset / IterablePackingDataset（packing）
  EncodePreprocessor（纯流式即时编码）
```

### 2.3 Loss 计算（`seq2seq_trainer.py` `compute_loss` L125）

```python
# L125-129
loss = template.compute_sft_loss(...)     # 常规：交叉熵，assistant 部分算 loss
# DFT / channel / loss_scale 场景:
loss = per_token_loss_func(...)           # trainers/utils.py L174（非 SP）
loss = per_token_loss_func_sp(...)        # L142（SP，分块 CE + GatherLoss）
```

`template.compute_sft_loss`（`swift/template/base.py`）的核心：

```text
labels 中非 assistant 部分为 -100（不参与 loss）
可选 loss_scale 逐 token 加权
支持 padding_free（cu_seqlens）+ sequence_parallel
```

### 2.4 训练循环（HF Trainer 标准 + SwiftMixin）

```text
training_step (seq2seq_trainer.py L231)
  ├─ template.forward_context（管理 SP / grad ckpt）
  ├─ compute_loss → loss
  ├─ accelerator.backward
  └─ optimizer.step + lr_scheduler.step
保存: SwiftMixin._save (mixin.py L416) → checkpoint-{step}
```

---

## 3. 预训练（PT）实现详解

### 3.1 与 SFT 的关系

`SwiftPretrain` **不重写任何方法**，仅把 `args_class` 改为 `PretrainArguments`（pretrain.py L11-16）。差异全在参数层：

| 参数 | 说明 |
|------|------|
| `truncation_strategy='split'` | 预训练长文档切分 |
| 无 system / 无模板对话 | 纯文本续训 |
| `--dataset` 可为纯文本文件 | txt / jsonl（raw text） |

### 3.2 数据差异

- 预训练数据通常是无监督纯文本，`RowPreprocessor` 走 `TextGenerationPreprocessor`（`dataset/preprocessor/extra.py` L55）或直接原始文本；
- packing 更常用（把多段文本拼到 max_length）；
- loss 覆盖全部 token（不同于 SFT 只算 assistant）。

---

## 4. RLHF 实现详解

### 4.1 入口（`SwiftRLHF`，`swift/pipelines/train/rlhf.py` L23）

```text
SwiftRLHF._prepare_model_tokenizer (L117):
  额外加载:
    ref_model（KL 约束）
    reward_model（奖励模型）
    value_model（PPO）
    teacher_model（GKD / OPD-RL）
  GKD 跳过 ref_model
SwiftRLHF._get_trainer_kwargs (L226):
  传 ref/reward/value/teacher_model + reward_template + vllm_client + reward_funcs
  ↓
TrainerFactory 按 rlhf_type 选 trainer
```

### 4.2 GRPO 训练循环（`grpo_trainer.py`）

```text
GRPOTrainer.__init__ (L90)
  ├─ _prepare_algorithm_params (L2107)
  ├─ prepare_rollout (rollout_mixin L115)   # vLLM 引擎 / 异步生成 / 多轮调度
  ├─ _prepare_rewards (L2195)
  ├─ _setup_teacher (rollout_mixin L198)    # OPD-RL
  └─ _prepare_liger_loss (L2059)

训练循环:
  dataloader (RepeatSampler / steps_per_generation)
    ↓
  _prepare_inputs (L189)
    ├─ _rollout_samples (L215): encode_sample + _generate_completions
    ├─ _score_completions: 奖励
    ├─ _compute_advantages
    └─ 拆分 micro-batch + 缓冲复用
    ↓
  compute_loss (L862) → _compute_loss (L878)
    └─ _compute_loss_and_metrics (L948)      # 见 03 篇
    ↓
  backward + optimizer.step
```

### 4.3 关键实现点

| 点 | 位置 | 说明 |
|----|------|------|
| 生成批与训练批分离 | `generation_batch_size` vs `per_device_train_batch_size` | 一次生成多组，多次更新 |
| 缓冲复用 | `_buffered_inputs`（L169） | 复用生成的 completions |
| 分块 loss | `_compute_loss_chunked`（L1276） | 降峰值显存 |
| ref 同步 | `SyncRefModelCallback`（L151） | `sync_ref_model` 时周期同步 |
| 动态采样 | `_dynamic_sampling`（L644） | DAPO |
| 多轮 | `MultiTurnScheduler` | agent 场景 |

### 4.4 各类 RLHF trainer 的 loss 一句话

| trainer | loss 核心 | 文件 |
|---------|----------|------|
| DPO | sigmoid/ipo/hinge 等变体 + ref logp | `dpo_trainer.py` L211 |
| SimPO | CPO + simpo_gamma（rlhf_args 改写） | `cpo_trainer.py` L31 |
| ORPO | 策略梯度 + NLL（odds ratio） | `orpo_trainer.py` |
| KTO | KL_logps 区分 desirable/undesirable | `kto_trainer.py` L50 |
| RM | logsigmoid(chosen - rejected - margin) | `reward_trainer.py` L38 |
| PPO | policy loss + value loss（HF PPO） | `ppo_trainer.py` |
| GKD | 分块 JSD（beta 插值 KL） | `gkd_loss.py` L94 |
| GRPO | clip 系 loss（见 03 篇） | `grpo_trainer.py` L948 |

### 4.5 多模型的管理

```text
actor:      要训练的主模型
ref_model:  KL 约束（beta * KL）
reward:     nn.Module 走 rm_plugin；函数走 reward_funcs
value:      PPO 的 value network（HFPPO）
teacher:    GKD / OPD-RL / SDAR 的教师（_setup_teacher）
```

---

## 5. 三大任务的保存差异

| 任务 | 保存内容 | 说明 |
|------|----------|------|
| PT/SFT | 模型 + optimizer + scheduler + state | SwiftMixin._save |
| DPO 等偏好 | 模型 + ref（不保存 ref） | 同 SFT 框架 |
| GRPO | actor 模型（+ 可选 ref 同步） | `save_model` 只存 policy |
| PPO | 只存 policy（`self.model.policy`） | ppo_trainer.py L86 |
| GKD | student 模型 | teacher 不保存 |

---

## 6. 训练前数据形态对照

| 任务 | 数据行示例 | 编码产物 |
|------|-----------|----------|
| PT | `{"text": "The quick brown fox..."}` | 全 token 作为 labels |
| SFT | `{"messages": [{"role":"user",...},{"role":"assistant",...}]}` | assistant 部分作为 labels |
| DPO | chosen + rejected messages | chosen/rejected 两条编码 |
| KTO | messages + label(desirable) | 右移构造 KL 项 |
| GRPO | prompt messages | prompt 编码 + 生成 completions |
| RM | messages + chosen/rejected | 两路编码取 logits |

---

## 7. 一句话总结

```text
PT = SFT（仅参数差异，全 token loss）
SFT = Seq2SeqTrainer + template.compute_sft_loss + Lazy/Packing 数据
RLHF = SwiftRLHF（多模型加载）+ 各 rlhf_type trainer（GRPO 最复杂，rollout→reward→advantage→loss）
```

深入阅读：
- [03-优化算法.md](03-优化算法.md)：RL loss 数学。
- [10-算法配置与跨Worker数据流转.md](10-算法配置与跨Worker数据流转.md)：GRPO 参数与流转。
- [04-容错与恢复机制.md](04-容错与恢复机制.md)：保存/恢复。
