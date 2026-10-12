# Stage 2B 联合两项 Loss 版（1.0·TF Cosine + 0.1·FR Execution CE）报告 — 2026-10-12

**结论：COLLAPSE（latent 与题目无关）+ EXECUTION_BYPASS。** 接口和梯度的 smoke test 9/9 通过，训练数值稳定，Explicit 输出可用；但学到的是**每个 slot 一个固定常量向量**，Executor 也不读这些 latent。
- 用量：smoke 1 步 + 训练 50 步，约 20 分钟（含评估）。没有 alpha，没有第三项 loss，没有噪声或 RL，系数固定为 1.0 / 0.1。

## 设置
- **Student**：Stage-1 Qwen3-4B（冻结）+ 一套 LoRA r16（q/v/o）+ LaTER projector（新初始化）。
  - Stage-1 是全参 SFT，checkpoint 里没有 LoRA，所以沿用上一轮的等效 LoRA。
  - 数据顺序、lr 1e-4、每步 16 个训练单样本，都与上一轮 KD-only 2B 完全相同。
- **Teacher**：冻结的 2A step-300 原始 Z_T。它本来就在 4B embedding 槽空间、RMS 与 token 相同，所以没有新增 adapter。
- **Forward A（TF）**：`chat(Q)+<EXPLORATION>\n+[Z_T1..Z_T7]` 做一次因果 forward，slot i 的预测 = proj(slot i 前一个位置的 hidden)。
- **Forward B（FR）**：从 Q 自由递归生成 Z_S，保留梯度，然后算 E+A 的 CE（含 `</EXECUTION><|im_end|>`）。
- **Smoke test**（`joint_gates.json`）：
  - Z_T[j] 只影响 slot i>j，Z_T[7] 不作为输入；
  - 把 Student 自己的 latent 喂进 TF 路径，能复现自由递归（cos 0.9999999）；
  - FR 路径与 Teacher 缓存无关；loss 就是 mean(1−cos)；
  - CE 梯度能传到 z1；base 权重不变。

## 训练曲线（`train_metrics.csv`）
| step | 1−cos（TF） | 1−cos（FR） | Exec CE | 相邻 slot cos |
|---:|---:|---:|---:|---:|
| 1 | 0.997 | 1.000 | 0.264 | 0.77 |
| 10 | 0.738 | 0.750 | 0.280 | 0.88 |
| 25 | 0.551 | 0.555 | 0.250 | 0.95 |
| 50 | 0.471 | 0.474 | 0.189 | — |

**加权梯度**（单样本，projector / LoRA）：

| step | 1.0·cos | 0.1·exec |
|---:|---|---|
| 1 | 0.89 / 0.12 | **2.32 / 0.34** |
| 10 | 0.55 / 0.12 | 0.029 / 0.032 |
| 50 | 0.27 / 0.11 | **0.0035 / 0.030** |

一开始 exec 项的梯度更大；10 步后它到 projector 的梯度降了约 3 个数量级，之后基本只有 cos 项在推 latent。

## 48 道未见题（`probe_joint_step*.json`、`evaluations/summary_joint_step50.json`）
| 指标 | step 0 | 10 | 25 | 50 | 参照 |
|---|---:|---:|---:|---:|---|
| TF cos：paired / shuffled | 0.001 / 0.000 | 0.292 / 0.290 | 0.460 / 0.457 | **0.544 / 0.541**（差 0.003，CI [−0.009, 0.015]） | 均值 Teacher 的 cos 为 **0.684** |
| FR cos：paired / shuffled | −0.001 / −0.002 | 0.281 / 0.280 | 0.456 / 0.455 | **0.541 / 0.540**（差 0.001，CI [−0.012, 0.012]） | |
| Z_S 跨题 cos | 0.835 | 0.962 | 0.996 | **0.996** | Teacher 0.465 |
| 跨题方差 / Teacher | 0.309 | 0.071 | 0.008 | **0.007** | |
| 相邻 slot cos | 0.81 | 0.90 | 0.94 | 0.81 | Teacher 0.41 |

- 逐 slot 看，TF 与 FR 几乎相同（step 50：slot1–8 为 0.71/0.88/0.46/0.46/0.34/0.42/0.53/0.55）。**即使看得到真实的前序 Teacher latent，预测也不随输入变化**，所以问题不在自由递归的误差累积，而是 projector 输出了每个 slot 一个固定常量。
- **Exec CE 干预**（step 50，与 self 的差）：crossq −0.0000，mean −0.0001，zero 0.0000，no-latent −0.0001，量级都是 1e-4。也就是 **Executor 不看 latent**。2A reader 读 Z_S 时，crossq−self = 0.001。

## 自由生成（step 50，greedy，2048 token）
- **held-out 8 题**：self 的正确率 7/8，8/8 正常以 `</EXECUTION><|im_end|>` 结束，Explicit 格式保持得很好。
- **但换成别的 latent，方法基本不变**：
  - 自动方法标签串：self = BBBABBAA，crossq = BBBABBAA，mean = BBBABBAA，zero = ABBABBAA；
  - 正确率：crossq 6/8，mean 7/8，zero 6/8；
  - 外部 Teacher A/B 切换方法 0/8，4 格 Method Choice 两向都正确 0/8。
- **训练 Choice 对 8 题**：self 正确 5/8，Teacher 切换 0/8。
- 原文见 `evaluations/generations_joint_step50.md`。

## 回答第 8 节的四个问题
1. **Learned latent？否。** paired 与 shuffled 没有差别；连“均值 Teacher”基线（0.684）都没达到。
2. **Free-running works？** 能稳定生成，TF 与 FR 的差距只有 0.003，但前提是 latent 根本不依赖输入；跨题方差只有 Teacher 的 0.7%。
3. **Executor works？是。** held-out 7/8 正确，格式完好，CE 从 0.26 降到 0.19。
4. **No trivial bypass？否。** 替换或去掉 latent，CE 只差 1e-4 量级，自由生成的方法不变。

## 诊断（7.3 中列出的可能原因逐条对照）
- **同题多 Plan 冲突**：排除。训练单样本每题只有一条 Plan。
- **空间或位置错配**：排除。Z_T 与 Z_S 在同一槽空间，TF 位置经过逐格泄漏检查。
- **TF 正常、FR 误差累积**：排除。TF 本身就和题目无关，并且 TF≈FR。
- **latent 被 detach、Executor 读不到**：排除。step 1 时 exec 梯度很大（projector 2.32），8 个 slot 都影响首 token。
- **某项梯度过度主导**：部分成立。10 步后 cos 项主导，exec 项衰减了约 3 个数量级。但更主要的是下一条。
- **学到常量均值而不是题目特异的表示**：**成立，是主因。**
  - 与上一轮 KD-only 2B（MSE）现象一致，只是 cosine 降得更快、更快到达常量解。
  - 上一轮只读探针已证明：projector 的输入 h_seed 里有可线性提取的 Z_T 题目信息（ridge 在未见题上 correct−shuffled 0.29，98% 为正）。所以这是优化路径的问题：这一步数和 lr 下，梯度先走“每个 slot 输出 Teacher 均值方向”的捷径。
  - latent 变成常量以后，CE 对 latent 的梯度也随之消失（一旦 latent 不携带信息，Executor 也就没有理由去读它），bypass 是坍缩的结果。

## 下一步（只给一个方向，需要你批准）
**先让 projector 一开始就是题目相关的，再跑这套两项 loss，系数不变。** 做法：用训练集上的闭式 ridge 解（h → Z_T，上一轮在未见题上已验证有效）初始化 projector 的映射。进入联合训练前设一个门槛：未见题上 paired−shuffled cos 的 CI 不跨 0，且跨题方差不低于初始值。不改 loss 形式，不加第三项 loss，不加 alpha。

## 文件
- `joint_gates.json`、`train_joint.jsonl`、`train_metrics.csv`、`probe_joint_step{0,10,25,50}.json`。
- `evaluations/summary_joint_step50.json`、`rows_joint_step50.jsonl`、`generations_joint_step50.md`。
- `checkpoints/joint_step{0,10,25,50}.pt`。
- 代码：`later/scripts/stage2t/joint_2b.py`（评估复用 `three_step.py --mode eval`）。
