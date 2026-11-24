# 离线对战与模型评估 · Offline evaluation

[English home](../README.md) · [Models / 模型](../checkpoints/README.md) · [Checks / 检查](validation.md)

本页说明如何在**本地离线引擎**上可靠地比较模型。所有对局都在用户自己的 VM/模拟器中运行，不连接任何游戏服务器。

This page describes how to compare checkpoints reliably on the local offline engine. Nothing here connects to game servers.

## 为什么需要新的评估流程

`evaluate.py` 适合快速对比两个模型，但存在三个统计问题：

1. **卡组与模型混淆**：A 始终使用 `deck0`、B 始终使用 `deck1`，卡组强弱会被算到模型头上。
2. **种子不配对**：交换上下方的两局使用不同种子，方差无法相互抵消。
3. **没有置信区间**：`--games 2` 的胜率没有统计意义；历史评测使用的“IL 15 秒强制出牌”协议也无法复现。

The quick evaluator confounds deck strength with model strength, does not pair seeds across side swaps, reports no uncertainty, and could not reproduce the historical forced-opening protocol.

## 工具一览

| 工具 | 用途 |
| --- | --- |
| `python -m native_runner.training.v4.tournament` | 多模型循环赛、卡组池、镜像赛程、断点续跑、SPRT 晋级门 |
| `python -m native_runner.training.v4.evaluate --force-first-action-tick 300` | 原有两模型评测，新增强制首动作选项与 `report.json` |
| `python -m native_runner.training.v4.eval_stats matches.jsonl ...` | 离线重算任意 `matches.jsonl` 的统计报告（无需 PyTorch 和引擎） |

## 镜像赛程 (mirrored blocks)

每个“块”使用**同一个种子**，从卡组池中抽两套卡组 X、Y，打 4 局：

| 局 | A 的位置 | A 的卡组 | B 的卡组 |
| --- | --- | --- | --- |
| 1 | owner 0 | X | Y |
| 2 | owner 1 | X | Y |
| 3 | owner 0 | Y | X |
| 4 | owner 1 | Y | X |

块内的上下方优势与卡组优势完全抵消，模型差异是剩下的唯一系统性因素。`--fixed-deck-a` 用于专精模型评测：A 固定卡组，每块 2 局（交换上下方）。

## 使用示例

```powershell
./native_runner/start_offline.ps1

# 先预览赛程，不需要引擎
.\.venv\Scripts\python.exe -m native_runner.training.v4.tournament `
  --checkpoint General=./checkpoints/General/checkpoint-step-00000460.pt `
  --checkpoint IL=./checkpoints/IL/checkpoint-step-00029396.pt `
  --checkpoint Hog2=./checkpoints/2_6hog_expert/hog26-specialist2.pt `
  --decks ./native_runner/training/v4/focus_decks_20.json `
  --blocks-per-pair 24 --output ./evaluations/league-v1 --dry-run

# 实际运行（去掉 --dry-run）。中断后用同样的命令重跑，已完成的局会被跳过。
```

候选模型晋级门（只适用于两个模型）：

```powershell
.\.venv\Scripts\python.exe -m native_runner.training.v4.tournament `
  --checkpoint candidate=./runs/ppo/checkpoint-step-00000600.pt `
  --checkpoint baseline=./checkpoints/General/checkpoint-step-00000460.pt `
  --decks ./native_runner/training/v4/focus_decks_20.json `
  --blocks-per-pair 250 --sprt 0 30 --output ./evaluations/gate-600
```

`--sprt 0 30` 检验 H0：Elo 差 = 0，H1：Elo 差 = 30（α = β = 0.05）。只在完整块的边界上检查并提前停止，`report.json` 的 `stopped_by_sprt` 给出结论。

卡组池可以是 `focus_decks_20.json`（RoyaleAPI 卡牌键）或下面的显式格式：

```json
{"decks": [{"name": "hog26", "cards": [26000021, ...], "form_availability": [0, 1, ...], "tower_troop_id": 159000000}]}
```

## 输出

| 文件 | 内容 |
| --- | --- |
| `identity.json` | 模型哈希、采样方式、强制首动作设置、基础对局配置、种子 |
| `schedule.json` | 卡组池与完整赛程（每局的 `game_id`、块号、位置、卡组、种子） |
| `matches.jsonl` | 每局一行：胜者、终局、动作数、拒绝数、首动作 tick、满费 tick、强制动作数、耗时；失败局为 `status: "error"` |
| `report.json` / `report.txt` | 统计报告（见下） |

`report.json` 包含：

- `pairs`：每对模型的胜/平/负、得分（平局算 0.5）、**Wilson 95% 置信区间**、Elo 差及其区间、可选 SPRT 结果。
- `ratings_elo`：所有模型的 **Bradley–Terry** 评分（以 0 为中心）。
- `decks` / `matchups`：每个模型在每套卡组上的得分，用于分析**跨卡组泛化**。
- `owner0_seat`：owner 0 座位的得分；显著偏离 0.5 说明引擎或观测存在座位不对称。
- `behaviour`：平均出牌数、首次出牌 tick、满费（≥10 圣水）停留 tick。历史经验表明胜率升高可能伴随更被动的打法，这些指标用来同时检查强度和行为。
- `status_counts`：`ok` / `truncated` / `error` 计数。截断和失败局**不计入**胜负统计。

## 解读规则

- 报告胜率时必须同时给出局数、对手、卡组池、解码方式（argmax / `--sample`）和是否强制首动作。
- 95% 区间宽度约为 `±1/√n`：200 局约 ±7%，800 局约 ±3.5%。区间重叠时不要宣称“更强”。
- 观察到的 Elo 差会被卡组/座位噪声压缩（逻辑函数的非线性），不同卡组池之间的 Elo 不能直接比较。
- `error` 或 `truncated` 比例超过 1% 时，先排查引擎问题，再看胜负。

## 已验证与未验证

- **已验证（模拟引擎）**：赛程平衡、续跑、失败隔离、SPRT 在块边界停止、统计量数值、20 套 focus 卡组都能生成合法 `MatchConfig`。见 `native_runner/tests/test_training_v4_eval_stats.py` 与 `test_training_v4_tournament.py`。
- **未验证（需要真实离线引擎）**：在 MuMu/AVD 上的完整对局吞吐、引擎异常后 `BattleEnvV1` 能否在下一局 `reset` 中恢复。若不能恢复，`--max-errors` 会在连续失败后停止运行，重启引擎后用相同命令续跑即可。
