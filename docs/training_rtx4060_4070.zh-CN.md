# RTX 4060 / 4070 单卡训练指南

[中文首页](../README.zh-CN.md) · [训练命令](training.md) · [训练历程](training_history.md)

本文面向一台装有 RTX 4060（8 GB）或 RTX 4070（12 GB）的台式机，说明哪些训练能在单卡上跑、该怎么配参数，以及本分支新增的省显存与提速选项。原项目的 PPO 在 8 张数据中心 GPU 和 1,528 场并发对局上完成，单卡不可能复现那个规模；单卡适合做模仿学习（IL）、在已有检查点上微调、评估，以及小规模 PPO 冒烟。

> 每条结论都标了来源：**[已验证]** 表示有单元测试，或在 Kaggle 免费 Tesla T4（16 GB）或云端 CPU 上实测过；**[待实验]** 表示推算或假设，需要你在 4060/4070 上用下文的基准脚本确认。显存占用基本与 GPU 型号无关，所以 T4 的显存数字可以直接用来估 4060/4070；速度则不能直接套用。

## 1. 先看瓶颈在哪里

| 环节 | 主要消耗 | 单卡上的情况 |
| --- | --- | --- |
| 回放重建 → IL 缓存 | CPU + 安卓模拟器中的原生引擎 | 与 GPU 无关；一次性工作，可以慢慢跑 |
| IL 训练 | GPU 显存（激活）+ 逐时间步 Python 循环 | **单卡主战场**，本分支的优化集中在这里 |
| PPO 采样 | CPU 核数、KVM 模拟器、引擎数量 | 瓶颈是模拟，不是 GPU |
| PPO 更新 | GPU 显存 | 需要把 lane microbatch 调小 |
| 评估 / 本地推理 | 很小 | CPU 也能跑（README 已说明） |

模型本身很小：生产模型 **12,053,037 个参数 [已验证]**。FP32 权重 + 梯度 + AdamW 两个动量约 0.2 GB，加上 CUDA 上下文约 0.5 GB。真正占显存的是 TBPTT 激活，它与「每次前向的序列数 × 时间步数」成正比。

### 每帧激活占用 [已验证：Kaggle T4 + CPU]

用 `tools/benchmark_training.py`（把测试夹具的观测填充到生产容量：48 个战斗组、160 个子实体、32 个事件、6 个候选动作）在 4 核 CPU 上测得：

| 配置 | 每次更新帧数 | 峰值内存 | 吞吐（帧/秒） |
| --- | --- | --- | --- |
| FP32，B=16，T=16 | 256 | 5.8 GB | 42.8 |
| BF16，B=16，T=16 | 256 | 3.9 GB | 75.1 |
| BF16，B=32，T=16，micro=1 | 512 | 6.5 GB | 66.1 |
| BF16，B=32，T=16，micro=4 | 512 | **2.8 GB** | 74.0 |
| BF16，B=32，T=16，micro=8 | 512 | 1.9 GB | 53.6 |

去掉约 0.9 GB 的进程基线后，CPU 上 BF16 约 11 MB/帧，FP32 约 19 MB/帧。

**Kaggle Tesla T4 实测（T=32，PyTorch 2.11）[已验证]**：

| 配置 | 每次前向帧数 | 峰值 allocated | 峰值 reserved | 吞吐（帧/秒） |
| --- | --- | --- | --- | --- |
| FP16，B=8 | 256 | 3.3 GB | 3.6 GB | 168 |
| FP16，B=16 | 512 | 6.5 GB | 7.3 GB | 311 |
| FP16，B=24 | 768 | 9.6 GB | 11.9 GB | 417 |
| FP16，B=32 | 1024 | 12.7 GB | 14.6 GB | 510 |
| FP16，B=48 | 1536 | 显存溢出 | — | — |
| FP16，B=48，micro=2 | 768 | 9.7 GB | 10.3 GB | 420 |
| FP16，B=48，micro=3 | 512 | 6.6 GB | 7.0 GB | 315 |
| FP16，B=48，micro=6 | 256 | 3.5 GB | 3.8 GB | 177 |
| FP32，B=8 | 256 | 5.5 GB | 6.5 GB | 157 |
| FP32，B=16 | 512 | 10.9 GB | 13.0 GB | 226 |

结论：

- **混合精度约 12.3 MB/帧，FP32 约 21 MB/帧**（按 allocated 计）。reserved 比 allocated 多 5–25%，并且帧数越多，多出的比例越大。BF16 与 FP16 的显存占用相同（T4 上 B=16 时 BF16 为 6.6 GB）。
- **微批次确实把显存压到「每次前向的帧数」对应的水平**：B=48 整批放不进 16 GB，拆成 2/3/6 份之后分别用 10.3、7.0、3.8 GB。
- **吞吐强烈依赖每次前向的序列数**：从 8 条到 32 条，吞吐从 168 提高到 510 帧/秒，原因是逐时间步的 LSTM/解码循环在小批量时受内核启动开销限制。所以微批次是**用速度换显存**：同样 48 条序列，micro=6 的速度只有 micro=2 的 42%。

## 2. 推荐配置

缓存默认每个分片 4 场回放 = 8 条己方序列，所以 `CR_IL_SHARDS_PER_RANK=S` 时每张卡每次更新有 `8×S` 条序列；`--time-steps` 默认 32。原作者的脚本默认值是 8 卡、每卡 6 个分片（每卡 48 条序列），单卡想要接近这个有效批量，就用微批次。

| 显卡 | 每次前向的帧数上限（估算） | 建议 | 每次更新 |
| --- | --- | --- | --- |
| RTX 4060 8 GB | 约 384 帧（12 条 × 32 步，约 5.5 GB reserved） | 保持原批量：`CR_IL_SHARDS_PER_RANK=6`、`CR_IL_MICRO_BATCHES=4`；求速度：`S=1`、不拆分 | 48 条序列、每次前向 12 条；或 8 条序列、不拆分 |
| RTX 4070 12 GB | 约 640 帧（20 条 × 32 步，约 9 GB reserved） | 保持原批量：`S=6`、`CR_IL_MICRO_BATCHES=3`；求速度：`S=2`、不拆分 | 48 条序列、每次前向 16 条；或 16 条序列、不拆分 |
| 显存仍不够 | — | 增大 `CR_IL_MICRO_BATCHES`，或 `--time-steps 16` | 时间步变短会改变 TBPTT 截断长度，属于超参数改动 |

这些上限是用 T4 实测的 12.3 MB/帧，加上显示输出和驱动占用的 1–2 GB 余量**推算的**。T4 上 B=16 实测 reserved 7.3 GB，对 8 GB 的 4060 太紧，所以 4060 建议每次前向 12 条。到 4060/4070 上的实际占用仍然**待实验**，正式训练前请先跑第 4 节的基准，把 `cuda_peak_reserved_mb` 控制在显存的 85% 以内。

其他设置：

- **混合精度用 BF16（默认）**，显存与 FP16 相同。Ada 架构（40 系）原生支持 BF16，不需要损失缩放。`--autocast float16` 现在会自动启用 `GradScaler` [已验证：选择逻辑有单元测试]，但没有理由在 40 系上优先用 FP16。
- **学习率**：微批次不改变有效批量，所以沿用原学习率。若你缩小了总序列数（例如只用 `S=2`），梯度噪声会变大，可以考虑把学习率从 3e-5 降低 [待实验]。
- **数据预取**：`CR_IL_PREFETCH_GROUPS=1` 会在后台线程加载下一组缓存分片，与当前组的 GPU 训练重叠 [已验证：顺序和异常传递有单元测试；实际节省的时间待实验]。代价是主机内存多存一组分片，16 GB 内存的机器先观察 `cache_load_s_rank0` 和内存占用再决定。
- **线程数**：`CR_IL_TORCH_THREADS` 保持 1–4；GPU 训练时 CPU 线程主要用于数据整理。

## 3. 运行方式

### Linux 或 WSL2（推荐）

IL 启动脚本是 bash，默认 NCCL 后端。Windows 用户最省事的做法是在 WSL2 里装 CUDA 版 PyTorch，单卡 NCCL 在 WSL2 中可用 [待实验：取决于驱动版本]。

```sh
export IL_WORLD_SIZE=1
export CR_IL_SHARDS_PER_RANK=6
export CR_IL_MICRO_BATCHES=3        # 4060；4070 用 2
export CR_IL_PREFETCH_GROUPS=1
bash native_runner/training/v4/launch_stateful_il.sh \
  --index /data/cache/index.json --train-tags /data/train-tags.txt \
  --output-dir runs/il-4060 --epochs 1
```

### 原生 Windows

原版训练器写死了 `backend="nccl"`，而 Windows 版 PyTorch 不带 NCCL，所以会直接失败。本分支新增 `--dist-backend auto`（默认）：有 NCCL 时用 NCCL，否则用 Gloo [已验证：2 进程 Gloo DDP 的梯度与单进程一致]。在 PowerShell 中：

```powershell
$env:USE_LIBUV = "0"   # 部分 Windows 版 torchrun 需要 [待实验]
python -m torch.distributed.run --standalone --nproc-per-node=1 `
  -m native_runner.training.v4.train_imitation_cache `
  --expected-world-size 1 --shards-per-rank 6 --micro-batches 3 --prefetch-groups 1 `
  --torch-threads 2 --no-validate `
  --index D:\cache\index.json --train-tags D:\cache\train-tags.txt --output-dir runs\il-4060
```

### 需要关注的日志字段

`train-metrics.jsonl` 的 `startup` 行现在会记录 `micro_batches`、`prefetch_groups`、`dist_backend`、`grad_scaler`。每组结束的 `group_complete` 行里：

- `cuda_peak_reserved_bytes`：接近显存上限就增大微批次。
- `owner_frames_s`：吞吐。
- `cache_load_s_rank0`：开启预取后表示「等待数据的时间」，应接近 0；明显大于 0 说明磁盘或解压跟不上。

## 4. 在你的显卡上做基准测试

基准脚本不需要游戏引擎和缓存，几分钟就能扫完。没有本地 GPU 时，可以把 [tools/kaggle/firstlight-gpu-benchmark.ipynb](../tools/kaggle/firstlight-gpu-benchmark.ipynb) 导入 Kaggle，用免费 GPU 运行；本文的 T4 数据见 [docs/benchmarks/kaggle-t4-2026-11-24.jsonl](benchmarks/kaggle-t4-2026-11-24.jsonl)。

```sh
# 找出单次前向能容纳多少条序列
python tools/benchmark_training.py --device cuda --autocast bfloat16 --time-steps 32 \
  --sweep 8,16,24,32,48 --json runs/bench.jsonl

# 比较同一有效批量下不同微批次数的速度
for m in 1 2 3 6; do
  python tools/benchmark_training.py --device cuda --autocast bfloat16 --time-steps 32 \
    --batch 48 --micro-batches $m --json runs/bench.jsonl
done
```

显存溢出时脚本会输出 `"error": "out_of_memory"` 并继续下一档。夹具模型的词表比生产模型小，但计算主体（关系 Transformer、空间卷积、LSTM、解码器）的维度完全一致，所以显存和速度可以代表真实训练；真实数据中有效实体比例不同，结果会有小幅偏差。

我希望你在 4060/4070 上验证的假设：

1. ~~GPU 上每帧约 11–16 MB~~ → T4 实测 12.3 MB/帧（混合精度）[已验证]。
2. ~~每次前向序列数少时 GPU 吃不饱~~ → T4 实测从 8 条到 32 条吞吐提高 3 倍 [已验证]。4060/4070 的绝对速度和 BF16 的速度仍**待实验**：T4 不原生支持 BF16，实测 BF16 只有 FP16 的 39%，这个数字不代表 40 系。
3. **预取能把 `cache_load_s_rank0` 降到接近 0** [待实验，需要真实缓存]。

## 5. 单卡上的 PPO

PPO 采样需要 Linux、`/dev/kvm` 和运行 ARM64 库的安卓模拟器，每个引擎至少占一个 CPU 核，每个引擎 8 个对局槽位。一台 8–16 核台式机现实的规模大约是 4–12 个引擎、32–96 场并发对局，比原作者少一个数量级以上 [待实验]。建议：

- 从 `checkpoints/General/checkpoint-step-00000460.pt` 开始，不要从零开始训 PPO。
- 更新时显存与 `--lane-microbatch-size × --sequence-chunk-steps` 成正比。默认 96 × 48 = 4,608 帧，按 11 MB/帧推算约 50 GB，单卡一定放不下。4070 可以从 `--lane-microbatch-size 16`（768 帧，约 8.5 GB）开始，4060 用 8–12 [待实验]。
- 采样时推理已经用了 CUDA Graph 和 BF16，单卡显存足够，瓶颈在模拟器。
- 把 `--rollout-segment-seconds` 保持在 40，靠增加更新次数而不是加大单次采样来积累样本。

## 6. 本分支改动一览

| 改动 | 位置 | 验证 |
| --- | --- | --- |
| 微批次梯度累积 `--micro-batches`（每个动作头按全局样本数归一化，DDP 下除最后一个微批次外都用 `no_sync`） | `training/v4/microbatch.py`、`train_imitation_cache.py` | 单元测试：2/3/8 个微批次的梯度、最终 LSTM 状态和损失指标都与整批一致；模拟 2 个进程、真实 2 进程 Gloo DDP 均与单进程一致；12 项测试在 Kaggle 上同样通过；T4 实测显存按微批次数下降 |
| 后台预取缓存分组 `--prefetch-groups` | 同上 | 单元测试：顺序、重叠、异常传递 |
| 自动选择分布式后端 `--dist-backend auto` | 同上 | 单元测试 + Gloo DDP 测试 |
| FP16 自动启用 `GradScaler`（原代码 FP16 没有损失缩放，有下溢风险） | 同上 | 单元测试 |
| 从 `imitation_loss` 中抽出 `il_loss_masks` / `il_loss_counts`，不做前向就能算出各头的样本数 | `training/v4/imitation.py` | 单元测试：与 `imitation_loss` 的计数一致；原有测试全部通过 |
| 合成基准 `tools/benchmark_training.py` | `tools/` | 已在 CPU 上运行 |
| 启动脚本读取 `CR_IL_MICRO_BATCHES`、`CR_IL_PREFETCH_GROUPS` | `launch_stateful_il.sh`、`.env.example` | `bash -n` 通过，`test_local_config` 通过 |

默认值（`--micro-batches 1`、`--prefetch-groups 0`）与原行为一致；`--dist-backend auto` 在 Linux CUDA 上仍然选择 NCCL。

## 7. 已知风险

- **DDP 未使用参数 [推断]**：训练器构造 DDP 时没有设置 `find_unused_parameters`。测试夹具的 effect/mechanic 目录为空，这些编码器不进入计算图，2 进程 DDP 下与它们同一个桶的参数（例如 `card_encoder`）就不会被同步，梯度静默出错，开启 `find_unused_parameters=True` 后恢复一致。生产目录不为空，但如果某张卡的一个 TBPTT 片段里完全没有 active effect，`active_effect_encoder` 可能同样不参与计算。多卡训练时值得检查；单卡不受影响。
- 仓库原有的 `test_probe_core_json.py::test_core_native_json_preserves_nullable_fields_precision_and_bounds` 在本环境（Linux x86_64 + 系统 g++）中失败，输出长度 502,676 字节，预期 508,076 字节。它与本分支无关，在未修改的 `main` 上同样失败。
