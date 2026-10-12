# Stage 2C AnchorSim 双 Loss 10 步排错实验 — 一页报告（2026-10-12）

- **性质**：这是 AnchorSim-inspired 的最小版本，Teacher 固定不更新（没有 EMA），只用来定位问题，不是论文的完整复现。
- **起点**：Student 和冻结 Teacher 都加载 `/scratch/hz33227/e2c/stage2/outputs/stage2b_joint_two_loss_20261012/checkpoints/joint_step50.pt`（step 50，phase joint，sha256 前 16 位 9baecb0ebc6e7d56）。原文件没有被覆盖。
- **损失**：L = 1.0·JSD(sg p_T, p_with) − 1.0·JSD(p_with, sg p_blind)，只有这两项。
  - Teacher 的输入是在题目末尾附加“(Reference final answer, visible to the teacher only: $a*$)”，并读入 sg(z_S)。
  - blind 分支：同样 8 个 slot 填零向量。
  - 只在 E 的文本 token 上计算，按 Execution token 的序号对齐；三个分支的 label 逐一核对，完全一致。
- **数据**：训练用 16 条（train_singles 的前 16 条），评估用 16 道（val_singles[48:64]），与训练集和此前的 48 道 latent 验证题都不重叠。具体 ID 见 `run_meta.json`。
- **训练设置**：AdamW lr 1e-4，optimizer 重新初始化，clip 1.0，单 seed。

## 1. AUDIT：PASS（15/15，`audit.json`）
- **可训练参数**：Student 的 projector 13.1M、LoRA 9.7M；Teacher 为 0，backward 后 Teacher 也没有任何梯度。
- **数值**：4 条样本的 loss 都有限。有效 token 数为 1315 / 837 / 677 / 639。Student 和 Teacher 的 prompt 长度不同（例如 167 与 184），但三个分支的 label 完全一致。
- **分开 backward 的梯度范数**：

  | 项 | projector | LoRA |
  |---|---:|---:|
  | Distill | 2.75e-3 | 1.32e-2 |
  | Anchor | 9.5e-4 | 3.8e-3 |
  | anchor / distill | **0.35** | **0.29** |
  | 总 loss（有限） | 2.73e-3 | 1.49e-2 |

  起点处 p_with≈p_blind，JSD 只有约 1e-4，但因为没有 cos 和 CE 项，anchor 的梯度并不是可以忽略的量级。
- **latent 替换确实生效**：slot 里放的确实是 z_S；换成零或别题的 latent 只改变 slot 的 embedding，非 slot 位置完全不变；logits 的最大差约 8.5–9.0。
- **泄漏检查**：答案只出现在 Teacher 的 prompt 里，Student 的 prompt 里没有。

## 2. 10 steps：COMPLETED（10/10）
- 训练约 4 分钟，加上第 0/5/10 步的评估共约 7 分钟；峰值显存 43.6 GB；数值全程有限。

| step | distill JSD | anchor JSD（正值） | total | ‖g‖ projector | ‖g‖ LoRA |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.00089 | 0.00016 | +0.00074 | 0.0011 | 0.0065 |
| 3 | 0.00091 | 0.0069 | −0.0060 | 0.111 | 0.027 |
| 5 | 0.0024 | 0.027 | −0.025 | 0.041 | 0.031 |
| 10 | **0.0165** | **0.079** | −0.062 | 0.157 | 0.062 |

## 3. 第 0 / 5 / 10 步评估（同一批 16 题）

| 指标 | step 0 | step 5 | step 10 |
|---|---:|---:|---:|
| 正确 / 16 | 11 | 1 | **0** |
| 格式有效 / 16 | 16 | 11 | **2** |
| with/blind JSD（E token） | 0.00015 | 0.035 | **0.116** |
| self latent 的 NLL | 0.213 | 0.396 | **0.919** |
| 改成零 latent 的 ΔNLL | +0.0003 | −0.183 | **−0.704** |
| 去掉 latent 的 ΔNLL | −0.0003 | −0.183 | −0.704 |
| 换成别题 latent 的 ΔNLL | +0.00005 | −0.00001 | +0.010 |
| 训练样本 distill JSD | 0.0009 | 0.0036 | **0.0235** |
| z_S 跨题 cos | 0.996 | 0.992 | 0.991 |

- 去掉 latent 时位置整体前移 8 格，换成零 latent 时位置不变，两者结果几乎相同。
- 第 10 步生成的文本出现了重复退化，例如 “或 or或 or…”、“$ x $, $ x $…”。

## 4. 结论：Loss 可优化，但没有改善 Bypass，反而是负 JSD 的伪成功
- with/blind JSD 上升了约 800 倍，但这是因为 **latent 被学成了一个所有题共用的“干扰信号”**：
  - 换成零或去掉 latent，NLL 反而下降 0.70；
  - 换成别题的 latent 几乎没有影响（+0.01），z_S 跨题 cos 仍为 0.99。
- 也就是说，模型的确开始受 latent 影响，但影响是破坏性的，而且和题目无关。
- **Distill 没有拉住它**：起点 Teacher 与 Student 几乎相同（JSD 9e-4），因为答案提示对 teacher-forced 的 gold E 预测几乎没有改变。所以 distill 的信号太弱；从第 3 步起 anchor 占了主导，distill JSD 反而一路上升到 0.0235。
- 答题能力被破坏：正确 11/16 → 0/16，格式 16/16 → 2/16。
- 实现和梯度路径没有问题：审计全部通过，三个分支对齐，梯度有限，Teacher 冻结。

## 5. 下一步只建议一项（需要确认，不会自动启动）
**先修问题，不要延长到 30 步。** 在这两项 loss 上加回正确性约束，即 Stage 2B 原有的 0.1·Execution CE（与 with 分支在同样的 gold E 上计算），然后用同样的 16 条数据再跑 10 步。判定标准：with/blind JSD 上升的同时，self latent 的 NLL 不能变差，并且换成别题 latent 的 ΔNLL 要变成正值。

这一步改变了“只用两项 loss”的规定，所以需要你确认。

## 文件
- `audit.json`、`run_meta.json`、`train.jsonl`、`evaluations/eval_step{0,5,10}.json`、`rows_step*.jsonl`（含生成文本）。
- `checkpoints/stage2c_step{5,10}.pt`。
- 代码：`later/scripts/stage2t/anchorsim_2c.py`。
