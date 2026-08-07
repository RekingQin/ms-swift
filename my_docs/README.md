# ms-swift 框架源码深度解读（主题索引）

> 目标读者：后续需全面负责 ms-swift 框架底层、训练、并行与架构优化的开发者。
> 解读基线：当前仓库 `feat/swift-learning` 分支（ms-swift v4.0，SWIFT 全家桶：训练/推理/评测/部署/量化）。
> 说明：本系列按主题拆分为独立文档，每个主题可独立维护迭代。架构/流程图均使用 ASCII 字符绘制。
> 阅读建议：新手从 [00-新手代码阅读指南.md](00-新手代码阅读指南.md) 开始，按主线走完后再深入各专题。

## 主题文档

| 编号 | 主题 | 文档 |
|------|------|------|
| 00 | 新手代码阅读指南（目录地图 / 阅读路线）**← 新人从这里开始** | [00-新手代码阅读指南.md](00-新手代码阅读指南.md) |
| 01 | 核心运行流程（CLI / Pipeline / 训练主链 / 推理评测部署） | [01-核心运行流程.md](01-核心运行流程.md) |
| 02 | 核心数据结构（参数系统 / 消息 / 数据集 / RL 样本与批次） | [02-核心数据结构.md](02-核心数据结构.md) |
| 03 | 优化算法（GRPO 族 / 偏好学习 / Loss / 奖励系统） | [03-优化算法.md](03-优化算法.md) |
| 04 | 容错与恢复机制（checkpoint / resume / 分布式后端） | [04-容错与恢复机制.md](04-容错与恢复机制.md) |
| 05 | 性能优化设计（显存优化 / 序列并行 / Tuner / 优化器） | [05-性能优化设计.md](05-性能优化设计.md) |
| 06 | 自定义扩展方式（注册机制 / 数据集 / 模型 / 模板 / Tuner / 奖励） | [06-自定义扩展方式.md](06-自定义扩展方式.md) |
| 07 | Debug 方法与技巧（日志 / profiling / 单卡调试 / 测试） | [07-Debug方法与技巧.md](07-Debug方法与技巧.md) |
| 08 | 算子库构成与原理（序列并行算子 / loss_scale / 外部算子集成） | [08-算子库构成与原理.md](08-算子库构成与原理.md) |
| 09 | 后续优化 Todo 指引（结合业界进展） | [09-后续优化Todo指引.md](09-后续优化Todo指引.md) |
| 10 | 算法配置与跨 Worker 数据流转（RLHF 配置 / rollout / 分布式流转） | [10-算法配置与跨Worker数据流转.md](10-算法配置与跨Worker数据流转.md) |
| 11 | 预训练、SFT 和 RL 实现详解（三大训练任务深入） | [11-预训练、SFT和RL实现详解.md](11-预训练、SFT和RL实现详解.md) |
| 12 | 流式数据处理（streaming / packing / DataLoader / JSONL） | [12-流式数据处理.md](12-流式数据处理.md) |
| 13 | 设计与实现架构（总体架构 + PT/SFT/RL 三模式 ASCII 架构图） | [13-设计与实现架构.md](13-设计与实现架构.md) |

## 源码导航速查

- CLI 入口：`swift/cli/main.py`（`ROUTE_MAPPING` 子命令分发 + `cli_main()`）、各子命令薄壳 `swift/cli/{sft,pt,rlhf,infer,deploy,export,eval,merge_lora,sample}.py`
- Pipeline：`swift/pipelines/base.py`（`SwiftPipeline` 基类）、`swift/pipelines/train/{sft,pretrain,rlhf}.py`、`infer/`、`eval/`、`export/`、`sampling/`
- 训练器：`swift/trainers/{trainer,seq2seq_trainer,trainer_factory,patcher,mixin,utils}.py`
- RL 训练器：`swift/rlhf_trainers/`（`grpo/dpo/kto/cpo/orpo/ppo/reward/gkd`）
- RL 算法核心：`swift/rl_core/`（`advantage.py`、`grpo_algorithm.py`、`data.py`、`resample.py`）
- 参数：`swift/arguments/`（`base_args/` + `sft_args.py`、`rlhf_args.py`、`tuner_args.py`、`pretrain_args.py` 等）
- 数据：`swift/dataset/`（`loader.py`、`register.py`、`packing.py`、`indexed_dataset.py`）、`swift/dataloader/`
- 模板：`swift/template/`、损失：`swift/loss/`、奖励：`swift/rewards/`
- 模型：`swift/model/`（`model_meta.py`、`register.py`、`model_arch.py`）
- Tuner：`swift/tuners/`（`base.py`、`mapping.py`、`lora.py` 等）
- 推理引擎：`swift/infer_engine/`（`vllm/sglang/lmdeploy/transformers`）、rollout 调度：`swift/rollout/`
- 序列并行：`swift/sequence_parallel/`（`sequence_parallel.py`、`ulysses.py`、`zigzag_ring_attn.py`）
- 优化器：`swift/optimizers/`（`galore/`、`lorap.py`、`muon.py`、`multimodal.py`）
- Megatron：`swift/megatron/`（TP/PP/CP/EP/VPP，`convert.py` 权重互转）
- 回调：`swift/callbacks/`、指标：`swift/metrics/`、工具：`swift/utils/`
