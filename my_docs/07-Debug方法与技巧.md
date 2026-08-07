# 07 · Debug 方法与技巧（日志 / profiling / 单卡调试 / 测试）

> 返回索引：[README.md](README.md)
> 前置阅读：[00-新手代码阅读指南.md](00-新手代码阅读指南.md)
> 本文回答：训练报错了怎么排查？怎么只看一张卡？profiling 怎么开？ms-swift 的测试体系怎么跑？

---

## 1. 日志体系

### 1.1 统一 Logger（`swift/utils/logger.py` L55）

```python
logger = get_logger()       # 全局获取
logger.info_once(...)       # 去重打印（L39）
logger.warning_once(...)    # 去重告警（L47）
logger.info_if(msg, cond)   # 条件打印（L27）
```

关键行为：

- 日志级别由环境变量 `LOG_LEVEL` 控制（L66，默认 `INFO`）；
- 非 master 进程强制 `ERROR`（L103），避免多卡刷屏；
- `logger.root.handlers` 的 StreamHandler 被设为 ERROR（L82-84），修复 DDP 重复日志。

```bash
# 调试时打印更多日志
LOG_LEVEL=DEBUG swift sft --model Qwen3-8B ...
```

### 1.2 `logging.jsonl`（训练过程日志）

`swift/trainers/patcher.py` `ProgressCallbackNew`（L33）：

- `on_log`（L51）：主进程把每次 log 追加写入 `{output_dir}/logging.jsonl`；
- `add_train_message`（L17）：注入 `global_step/max_steps`、`elapsed_time`、`remaining_time`、`train_speed`。

```bash
# 训练中断后回看完整日志
tail -f output_dir/logging.jsonl
```

### 1.3 外部追踪

```bash
--report_to tensorboard    # 默认
--report_to wandb          # 需要 WANDB_PROJECT（默认 ms-swift）
--report_to swanlab
```

- `swift/utils/tb_utils.py` `read_tensorboard_file`（L10）：读 `events.out.tfevents.*` 并绘图；
- RL 场景 `rlhf_trainers/utils.py` `profiling_context`（L582-586）：主进程且 report_to 含 tracker 时 `wandb.log` / `swanlab.log`。

---

## 2. Profiling

### 2.1 计时 profiling（`swift/rlhf_trainers/utils.py`）

- `profiling_context`（L565）：上下文计时器，把耗时以 `profiling/Time taken: {Class}.{name}` 记录；
- `profiling_decorator`（L589）：装饰器包裹方法；
- `memory_time_profiling_context`（L391）：记录 GPU allocated/reserved/peak 内存 + 耗时，可 `reset_peak_memory_stats`（L460）。

大量用于 RL 训练器：

```text
grpo_trainer.py: _prepare_inputs (L188), _score_completions (L464), compute_loss (L861)
gkd_trainer.py: 各关键方法
rollout_mixin.py: 生成相关
```

### 2.2 性能回调（`swift/callbacks/perf_log.py`）

`PerfMetricsLogCallback`（L36）：估算 **MFU / TFLOPS** 性能指标，`device_flops_map`（L16）含各 GPU 理论算力。

---

## 3. 单卡调试

### 3.1 单设备模式（`swift/cli/utils.py` L5）

```python
def try_use_single_device_mode():
    # SWIFT_SINGLE_DEVICE_MODE=1 时:
    #   按 LOCAL_RANK 从 CUDA_VISIBLE_DEVICES 取对应设备
    #   重设 CUDA_VISIBLE_DEVICES 并强制 LOCAL_RANK=0
```

```bash
# 只跑第一张卡
SWIFT_SINGLE_DEVICE_MODE=1 swift sft --model Qwen3-8B ...
```

### 3.2 多卡/单卡自动切换

- `NPROC_PER_NODE` 设置时走 torchrun（`cli/main.py` `use_torchrun` L30）；
- 单卡时自动置 `NPROC_PER_NODE=1`、`RANK=0`（`torch_utils.py` L328）；
- `swift/utils/env.py` `get_dist_setting`（L27）读取分布式环境变量。

### 3.3 数据/内存检查

- `DataArguments.resume_from_checkpoint`（data_args.py L60）：调试时跳过损坏样本继续；
- `RewardTrainer.visualize_samples`（reward_trainer.py L74）：奖励模型打印样本表格到 wandb。

---

## 4. 参数校验（提前暴露错误）

ms-swift 在启动阶段做了大量校验，报错信息通常直接指向解决方案：

| 校验 | 位置 |
|------|------|
| `soft_overlong` 需 `soft_cache_length` | `rlhf_args.py` L362 |
| `advantage_reweight=rlsd` 必须有 `reward_funcs` | `rlhf_args.py` L598 |
| `dynamic_sample` / `scale_rewards='gdpo'` 需 `reward_funcs` | `rlhf_args.py` L692 |
| GRPO/GKD 用 vLLM 与 `device_map` 冲突 | `rlhf_args.py` L542/L752（提示设 NPROC_PER_NODE） |
| GRPO 无 reward 则报错 | `grpo_trainer.py` L126-127 |
| LISA 回调要求 full tuner | `callbacks/lisa.py` L15 |

调试建议：**先看启动阶段的 ValueError**，再进训练循环。

---

## 5. 指标与评估

### 5.1 评估指标（`swift/metrics/`）

```text
eval_metrics_map (mapping.py L10): acc / nlg / infonce / paired / reranker
```

- `EvalMetrics`（base.py L11）抽象基类：`compute_metrics`（L18）+ `preprocess_logits_for_metrics`（L21）；
- `AccMetrics`（acc.py L44）：token/seq 准确率；
- `MeanMetric`（utils.py L73）：`compute` 内 `dist.all_reduce(SUM)` 跨 rank 聚合；
- 选择：`--metric acc`，训练器在 `mixin.py` L1069 实例化并注入。

### 5.2 训练内置指标

GRPO 训练器记录的指标（`_prepare_metrics` grpo_trainer L2078）：

```text
reward/reward_mean、reward/reward_std、reward/frac_reward_zero_std
kl、entropy/mean、clip_ratio/low_mean、clip_ratio/high_mean ...
```

---

## 6. 测试体系

### 6.1 目录结构（`tests/`）

```text
tests/
  ├─ train/       # 训练测试（test_grpo.py、test_gkd.py、test_kto.py ...）
  ├─ infer/       # 推理测试
  ├─ eval/  export/  deploy/  sample/
  ├─ tuners/      # tuner 测试（test_swift_base.py 27KB）
  ├─ models/  hub/  megatron/  llm/  general/
  ├─ test_align/  app/
  ├─ run.py       # 统一测试调度（22KB，--case-model-info / NPU 检测）
  └─ test_utils.py
```

### 6.2 运行方式

```bash
make test          # Makefile L16 → .dev_scripts/citest.sh
python tests/run.py --case-model-info ...   # 手动跑指定 case
```

- 测试框架：**unittest**（`setup.cfg` 只配代码风格 isort/yapf/flake8）；
- CI：`.github/workflows/citest.yaml`。

### 6.3 把测试当文档

```bash
# 看某个功能怎么配置/使用
grep -n "rlhf_type" tests/train/test_grpo.py
```

---

## 7. 常见问题排查路径

| 现象 | 排查 |
|------|------|
| 训练没启动就报参数错 | 看启动阶段 ValueError（rlhf_args 校验） |
| loss 异常（NaN） | 看 `logging.jsonl` 的 kl/clip 指标 + `--log_entropy` 开 entropy 日志 |
| 显存 OOM | `memory_time_profiling_context` + `nvidia-smi` + 开 activation offload / grad ckpt |
| rollout 没更新权重 | 检查 vLLM 引擎 weight 更新路由（rollout server） |
| 多卡日志刷屏 | 默认非 master 只打 ERROR |
| 训练速度慢 | `PerfMetricsLogCallback` 看 MFU/TFLOPS |
| 恢复位置不对 | 看 `dlrover_latest.txt` / `trainer_state.json` |

---

## 8. 采样工具（数据调试）

```bash
swift sample --dataset my_data --sampler_type vanilla ...
```

`SwiftSampling`（`swift/pipelines/sampling/sampling.py` L18）：对数据集离线采样生成，用于在训练前检查 prompt/模板效果；支持 `data_range` 分片断点续采。

---

## 9. 一句话总结

```text
日志 → LOG_LEVEL + logging.jsonl + wandb/tensorboard
计时 → profiling_decorator / memory_time_profiling_context / PerfMetricsLogCallback
单卡 → SWIFT_SINGLE_DEVICE_MODE=1 / NPROC_PER_NODE
校验 → 启动阶段 ValueError
测试 → tests/run.py + unittest
```

深入阅读：
- [09-后续优化Todo指引.md](09-后续优化Todo指引.md)：debug 工具的改进方向。
- [12-流式数据处理.md](12-流式数据处理.md)：流式场景的调试注意点。
