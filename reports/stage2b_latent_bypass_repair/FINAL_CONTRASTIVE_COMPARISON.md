# Stage 2B Latent Bypass Repair：D0 / A（Anchor）/ B（Plan-aware）对照报告（2026-10-12）

**结论：A、B 都没有通过 10 步诊断。** 所有测试过的权重下，指标都与同样训练 10 步的 D0 无法区分。按预先写定的规则，A、B 都没有延长到 50 步，C 也没有运行。
- **D0（原 2-loss 继续训练 50 步）**：latent 的跨题方差略有恢复（0.0071→0.0195），但 paired−shuffled 的 CI 仍然跨 0；Bypass 没有改变；准确率保持 64/128。
- **Generator 仍然坍缩，不具备进入 Stage 3 的条件。**
- 说明：A 是 *AnchorSim-inspired anchor-only ablation*，不是完整的 AnchorSim（没有 answer-conditioned EMA teacher）。B 是我们自己提出的扩展，不是 AnchorSim 原文的方法。

## 0. 做了什么
- **起点**：`stage2b_joint_two_loss_20261012/checkpoints/joint_step50.pt`（projector、LoRA 和 AdamW 状态都在，没有被覆盖）。每个分支都独立从这个文件加载。
- **训练设置**：所有分支使用同一个单样本数据序（原训练的全局第 51–100 步，每步 16 条）、同样的 lr 1e-4 和 clip 1.0。
- **损失**：D0 = 1.0·TF cos + 0.1·FR exec CE。A、B 在此基础上各加一个显式的第三项，两个原系数都没动。
- **没有做的**：没有实现 LIVR、EMA teacher、RL 或加噪声；L=8、backbone、2A encoder、prompt、EOS、LoRA 配置、projector、CE mask 全部不变。
- **测过的全部配置**（每个都先跑 10 步诊断）：
  - A：λ_anchor = 0.01、0.05、0.1；
  - B：λ_plan = 0.02、0.05（τ=1，η=1，每步 2 对）；
  - D0 跑满 50 步。
- **单 seed**（seed 0）。没有任何正信号，所以没有复测其他 seed。
- **实现**（`later/scripts/stage2t/contrastive_2b.py`）：
  - A 的 blind 分支：同样 8 个 slot 填零向量，在 `torch.no_grad` 下计算；ids、位置、mask、targets 与 aware 分支完全相同。
  - A 的 T：Execution 目标的前 128 个 token，不含 Q、slot、padding 和 `</EXECUTION><|im_end|>`。
  - B 的 strategy span：E_A/E_B 的 token 最长公共前缀之后、`</EXECUTION>` 之前的部分。matched CE 算在完整目标上。L_plan 只更新 LoRA。

## 1. Gate（`audit_gate.json`、`test_contrastive.json`）
- **G0 全部通过。**
  - 表示空间：Z_T 和 Z_S 的 shape 都是 [·,8,2560]，每个 slot 的 RMS 都等于 token-embedding RMS（0.021953）。
  - 注入：aware 和 blind 的 ids、pos、att 完全相同，非 slot 位置的 embedding 也完全相同；aware 的 slot 里确实是 Z_S，blind 的 slot 全为 0。
  - 数值一致：`exec_forward` 与原 `Student.score` 一致（−0.317308 / −0.317308）。
  - JSD 本身：JSD(p,p)=0；扰动越大 JSD 越大（0.001 → 0.50）；JSD ≥ 0 且 ≤ ln2。
  - 梯度路径：在 −JSD 上做一步下降，JSD 会增大（0.096 → 0.123）；blind 分支没有梯度；anchor 梯度能传到 projector、LoRA 和 dz1。
- **B 的单测通过**：L_plan 对 projector 的梯度为 0、对 LoRA 大于 0；A/B 交换后 Δ_A 与 Δ_B 互换；两个 Z 相同时 Δ=0。base 权重和 2A 缓存都没有改变。16/16 通过。
- **G1 通过，但样本量不足。**
  - 27 对同题 A/B（训练 19 对，held-out 8 对；s1val-00471 因为两条是同一种方法而排除）。来自 bridge 数据：两条答案都经 math_verify 判为正确，方法差异和“执行确实按照计划”都经过词表检查加人工通读（见 `bridge/build_pairs.py`）。
  - 按 question 划分，各集合之间零重叠，没有用 AIME。
  - Teacher 可区分：同题 cos(Z_T^A, Z_T^B) 平均 0.72（0.54–0.84），跨题 0.49。
  - **不足：held-out 只有 8 题，低于任务书要求的 32 道。** 所以下面所有 A/B 结果都只是探索性的，不做显著性结论。
- **G2**：48 道 latent 验证题固定不变；8 道执行题只做回归检查。另外从 stage2_val 抽了 128 道独立未见题（去重后有 2316 道候选，seed 0），统一用 greedy、最多 2048 token、`\boxed` 加 math_verify 判分。

## 2. 结果

### 7.1 答题能力
| Run | 额外步数 | λ_anchor | λ_plan | 准确率（N=128） | 格式完整 | Gold CE（48 题，self） | tokens/s* | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Start-50 | 0 | – | – | 64/128 = 50.0% [40.6, 58.6] | 114/128 | 0.2318 | – | – |
| D0 | 50 | 0 | 0 | 64/128 = 50.0% [41.4, 58.6] | 111/128 | 0.2265 | 1063 | 23.8 GB |
| A | 10（诊断） | 0.01 / 0.05 / 0.1 | 0 | 未评估 | – | 0.2298 / 0.2299 / 0.2298 | 1326 / 935 / 1080 | 23.7 GB |
| B | 10（诊断） | 0 | 0.02 / 0.05 | 未评估 | – | 0.2298 / 0.2297 | 1692 / 1215 | 62.4 GB |
| C | 未运行 | | | | | | | |

- 同题 10 步时 D0 的 Gold CE 是 0.2298。
- 起点和 D0 的逐题对比：两者都对 51 题，只有起点对 13 题，只有 D0 对 13 题。
- \*吞吐只作参考：有些分支与其他任务共用同一块 GPU。

### 7.2 Collapse 恢复情况（48 道未见题，按题 bootstrap 95% CI）
| Run | TF paired | TF shuffled | TF gap [CI] | FR paired | FR shuffled | FR gap [CI] | 跨题 cos | var 比 |
|---|---:|---:|---|---:|---:|---|---:|---:|
| Start-50 | .5438 | .5407 | +.0030 [−.009, .014] | .5408 | .5399 | +.0009 [−.012, .012] | .9962 | .0071 |
| D0 +10 | .5632 | .5602 | +.0030 [−.011, .016] | .5596 | .5584 | +.0012 [−.013, .014] | .9963 | .0069 |
| D0 +25 | .5839 | .5796 | +.0043 [−.009, .017] | .5830 | .5806 | +.0024 [−.011, .014] | .9948 | .0097 |
| **D0 +50** | .6296 | .6199 | **+.0097 [−.006, .024]** | .6284 | .6224 | +.0061 [−.010, .021] | .9896 | **.0195** |
| A .01 +10 | .5632 | .5602 | +.0030 | .5597 | .5585 | +.0012 | .9963 | .0070 |
| A .05 +10 | .5632 | .5602 | +.0030 | .5596 | .5585 | +.0011 | .9962 | .0070 |
| A .1 +10 | .5632 | .5602 | +.0030 | .5595 | .5584 | +.0011 | .9962 | .0070 |
| B .02 +10 | .5631 | .5601 | +.0030 | .5598 | .5586 | +.0011 | .9962 | .0070 |
| B .05 +10 | .5632 | .5602 | +.0030 | .5596 | .5585 | +.0011 | .9962 | .0070 |

- 起点能复现原报告的数字（0.544/0.541、0.541/0.540、0.996、0.007）。
- 逐 slot、逐样本的分布在 `evaluations/probe_*.json`。

### 7.3 Bypass 与 Plan Faithfulness（分开报告）
| Run | aware↔blind JSD（T 段） | self↔crossq JSD | Zero ΔGold-CE | Cross-Q ΔGold-CE | 同题 matched−swapped logP：held-out 8 [CI]（两向都对） | 训练 19 对（两向都对） | A/B 生成切换 | 准确率 |
|---|---:|---:|---:|---:|---|---|---:|---:|
| Start-50 | 3.1e-4 | 7.1e-5 | +1.0e-4 | −2.0e-5 | −.0001 [−.0045, .0038]（1/8） | +.0012（6/19） | 0/8 | 7/8\*；64/128 |
| D0 +10 | 2.8e-4 | 7.4e-5 | −6.8e-5 | +1.0e-4 | −.0003（0/8） | +.0012（4/19） | – | – |
| D0 +50 | 2.9e-4 | 6.9e-5 | +1.9e-4 | −7e-7 | +.0004 [−.0035, .0037]（1/8） | +.0012（7/19） | 0/8 | 6/8\*；64/128 |
| A .01 / .05 / .1 +10 | 2.9 / 2.8 / 3.0e-4 | 7.6 / 7.3 / 7.5e-5 | ≤2.5e-4 | ≤1.2e-4 | −.0003（1 / 1 / 2 /8） | +.0011~.0012（4–5/19） | – | – |
| B .02 / .05 +10 | 3.1 / 2.8e-4 | 7.3 / 7.3e-5 | ≤1.6e-4 | ≤1.6e-4 | −.0001 / −.0005（1/8） | +.0015 / +.0013（4 / 6 /19） | – | – |

- 带 \* 的数字只有 8 题。
- “同题 matched−swapped”用的是 strategy-span 的平均 log p。由于 LCP 只有 2–29 个 token，它和完整 E 的结果几乎一样。人工审计过的 24-token 决策段（D-span）也一样接近 0：起点 held-out 为 −0.003（0/5），D0 为 +0.0001（1/5）。
- **生成结果（`cases.jsonl`）**：起点和 D0+50 下，注入 Z_T^A 与 Z_T^B 后，自动方法标签分别是 BBBABBAB 和 BBBABBAB，两者完全一样（图 4 的混淆矩阵是 [[2,6],[2,6]]）。D0+50 时，self、crossq、mean、zero 四种条件下的标签也完全相同。
- **训练时 B 的 rank loss 一直停在 ln2 附近**（0.6927–0.6944），Δ_A、Δ_B 只在 ±0.02 内波动。也就是说，**连训练用的那 19 对都没有拟合动。**

### 7.4 图
- `fig1_loss_grad_vs_steps.png`：各项 loss，以及加权后的分项梯度。
- `fig2_latent_collapse_vs_steps.png`：gap、var 比、跨题 cos 随训练步数的变化。
- `fig3_bypass_vs_accuracy.png`：Bypass 敏感度与准确率、Gold CE 的关系。
- `fig4_plan_following_2x2.png`：同题 2×2 似然，以及生成方法的混淆矩阵。

## 3. 为什么 A、B 没有效果（有直接测量）
- **A：负 JSD 在坍缩点几乎没有梯度。**
  - 起点上 p_aware≈p_blind，JSD 只有约 3e-4 nat。
  - JSD 在 p=q 附近是二阶的，梯度正比于 (p−q)，所以也接近 0。
  - 实测第 10 步、item 0 的加权梯度，λ=0.1 时：

    | 项 | projector | LoRA |
    |---|---:|---:|
    | anchor | 1.9e-4 | 1.1e-3 |
    | cos | 0.256 | 0.141 |

    anchor 大约只有 cos 的 1/1400（projector）和 1/130（LoRA）。λ=0.01 时还要再小 10 倍。
  - 所以 A 的三个权重下，训练曲线都与 D0 在 4 位小数内一致。要让 anchor 起作用，λ 必须比任务书的网格大两到三个数量级。这已经超出“最小修复”的范围，我没有自行尝试。
- **B：rank 的梯度并不消失（softplus′(0)=0.5），问题在于 Executor 对 Z_T^A 和 Z_T^B 几乎不加区分（Δ≈1e-3）。**
  - 能推动 Executor 的只有 LoRA 梯度：λ=0.05 时，rank 项为 0.0027，matched 项为 0.012，同一 batch 中 cos 项为 0.137。
  - 结果 10 步内训练对的 gap 没有变化（+0.0012 → +0.0013）。
  - 按设计，L_plan 不会更新 projector，因此 Student 的 latent 本来就不会因此改变。
- **D0 的轻微恢复**来自继续用原 2-loss 训练，与 A、B 无关：var 比 0.007 → 0.0195，TF gap +0.003 → +0.010（CI 仍跨 0）。

## 4. 按第 8 节预先写定的标准判定
- **A 组：FAIL。** latent 敏感度没有比 D0 提高：JSD 3e-4 不变，ΔCE 在 1e-4 量级。
- **B 组：FAIL（no improvement）。** same-Q 的 matched−swapped 似然差和生成切换都没有提高，训练对上也没有。
- **Generator 仍然坍缩**：var 比 0.0195，FR gap 的 CI 跨 0。**不能进入 Stage 3。**
- **证据强度**：held-out 只有 8 题，都是探索性结果；128 题准确率只用来确认答题能力没有退化。

## 5. 局限
- 只用了一个 seed；同题 A/B 的 held-out 只有 8 题；方法标签是自动的 n-gram 判定，人工复核需要用 `cases.jsonl`。
- A 和 B 只跑了 10 步，因为没有通过诊断。如果效应在更晚才出现，这次测不到。不过从梯度量级看，A 的三个权重相当于 D0。
- 128 题准确率用的是 left-padding 批量 greedy，与单条生成在数值上可能略有差别；但所有 checkpoint 用的是同一套协议。

## 6. 下一步（按决策树“两者都不改善”分支，只给一个建议，需要你批准）
数据和注入接口都已经通过 G0/G1。剩下没回答的问题是：**这套 LoRA 能不能学会读取 Teacher latent？还是只是权重太小？**
- **建议做一个最小的“能力诊断”**：保留 L0，只在 19 对训练对上把 λ_plan 提高到让 L_plan 成为主导项（例如 1.0），跑 10–20 步，只看训练对上的 matched−swapped gap 和 A/B 生成切换。这不是新方法，只用来判断“Executor 可读性”是否可以训练出来。
- **如果连训练对都拟合不动**，问题就在 Executor/LoRA 的读出能力或 Teacher 的语义，应该先回到 2A 或 Executor 一侧，而不是继续加 contrastive 的步数。
- A（负 JSD）在坍缩点的梯度结构上就不利，不建议继续调 λ_anchor。

## 7. 运行记录
- **代码**：分支 `exp/stage2b-latent-bypass-repair`（基于 15ed0ce），提交号见 git log；新增 `contrastive_2b.py`、`test_contrastive.py`、`contrastive_report.py`。
- **起点 checkpoint**：`.../stage2b_joint_two_loss_20261012/checkpoints/joint_step50.pt`（sha256 前 16 位见 `ck0.sha`）。
- **硬件**：作业 48907334，e4-1，2× NVIDIA B200。
- **峰值显存**：D0 和 A 约 24 GB，B 约 62 GB。
- **时间**：GPU wall-clock 约 10:28–11:00，约 32 分钟，含单测、全部训练和评估。
  - D0 50 步：671 s；
  - A 每个权重 10 步：105–149 s；
  - B 每个权重 10 步：107–149 s；
  - 128 题准确率：486–778 s。
- **所有测过的 λ**：anchor 0.01、0.05、0.1；plan 0.02、0.05。
- **没有异常退出**，没有出现 NaN 或 Inf（grads_finite 全为 true）。
