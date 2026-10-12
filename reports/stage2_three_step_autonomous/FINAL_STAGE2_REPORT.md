# Stage 2 三步版 Pilot 最终报告（2026-10-12）

**状态：KD_COLLAPSE（Stage 2B Gate），已按任务书 7.2 停止，没有运行 2C。**
- 工程部分 ENGINEERING_PASS：14/14 项硬性单测通过，Tiny Overfit 弱通过。
- Teacher 判定为 TEACHER_SEMANTICS_WEAK：重建层面合格，方法层面偏弱。
- 预算：optimizer step 用了 50/160（Tiny 10 + 2B 40）；wall-clock 约 35 分钟。
- 没有加噪声，没有用 Set Matching，没有 RL，没有新增 Encoder/LoRA；旧 checkpoint 一个都没覆盖；没有 push。

## 1. 做了什么
- **Student**：Stage-1 Qwen3-4B（冻结）+ 一套共享 LoRA（r16，q/v/o）+ 新初始化的 LaTER Recurrent Projector。只看 Q，确定性地递归生成 z1..z8，写进 `<EXPLORATION>` 的 8 个 embedding 槽，再生成 `</EXPLORATION><EXECUTION>` + E。
- **Teacher**：冻结的 2A step-300 Compressor(P) 的原始输出。它本来就逐向量校准到 4B token-embedding RMS，而且 2A reader 就是这样读它的，因此和 Z_S 在同一接口上，不需要任何 adapter。2B step-50 adapter 没有使用，因为它会造成训练题泄漏。
- **2B**：40 步，每步 16 个训练单样本（每题一条 Plan），只用 KD。
- **评估**：48 道未见题（val singles）+ 8 对 held-out + 19 对训练对。指标包括 KD correct/shuffled、latent 几何、Exec CE 干预（self / crossq / mean / zero / teacher / no-latent）、方法 span 偏好，以及 2A reader 读 Z_S 的结果。

## 2. 结果总表

| 问题 | 证据 | 结果 / 结论 |
|---|---|---|
| 2A Teacher latent 在未见题上真有 Plan 信息吗？ | 2A 报告：打乱 Z 的 CE 比正确 Z 高 0.40，300/300 为正；同题 A/B 的 CE held-out 8/8。本轮 reader 自由生成：训练对两边都对 10/20（自动判定），held-out 人工判定 3/8 清楚切换 | **有 Plan 信息，但方法层面偏弱**（TEACHER_SEMANTICS_WEAK）。只看方法 token 时 Δ 只有 0.04 |
| 2B Q→Z 学成了题目相关 latent，还是平均向量？ | 未见题 KD correct 2.002→1.598→**1.168**，shuffled 2.004→1.599→**1.169**（差 0.001，CI 跨 0）；均值 Z 的 KD 为 1.163；Z_S 跨题 cos 0.835→0.911→**0.987**（Teacher 0.465）；跨题方差比 0.31→0.17→**0.024**；reader CE 中 crossq−self = 0.001 | **平均向量，判定 KD_COLLAPSE。** KD 的下降全部来自对齐 Teacher 的公共方向，题目差异反而被压掉；KD 仍高于“直接输出 Teacher 均值”的基线 0.555 |
| 2C Control 的 Execution CE 是否仍然 Bypass？ | 2C 没有运行。2B 第 40 步：未见题上 Exec CE 中 crossq−self = −0.0001，mean−self = −0.0001，no-latent−self = +0.0008，量级只有 1e-4 nat | 没有 2C 数据。按 2B 的现状，Executor 不依赖 Z_S，因为各题的 Z_S 本来就几乎相同 |
| 2C Choice 的改进是否来自 Method Choice？ | 未运行 | **无法回答**（被 Gate 阻断） |
| Method Choice 的实际有效梯度与训练频率如何？ | 只读审计，在 2B step 40 用 2C 的前 4 个 batch（8 单样本 + 2 对）：‖∇L_main‖ 中位数 0.40，‖∇L_choice‖ 中位数 1.13；Choice_self 0.83–1.01，几乎全部经过 LoRA；Choice_ext 0.10–0.98，只经过 LoRA；cos(∇main, ∇choice) 在 −0.05 到 0.14 之间 | **这个设置下 Choice 梯度并不弱**（与上一轮 1/170 的结论不同），而且与主 loss 近似正交。按预注册规则 λ_choice = 0.176，但没有用上 |
| 外部 Z 与自主 Z 的方法控制是否分别成立？ | 外部（未训练 Executor，raw Z_T）：4 格两向都正确 held-out 0/8、训练 0/19，m(Z_TA)−m(Z_TB) 均值 −0.009。自主：held-out 上 \|pref(self)−pref(crossq)\| = 0.008，\|pref(self)−pref(mean)\| = 0.007；换成同题另一 Plan 的 Teacher Z 后，往默认方向为正的只有 1/8 | **两者都不成立。** 外部 Z 需要 Choice_ext 训练才可能起作用，而这一步没跑到 |
| 最终 Q→Z→Explicit CoT→Answer 的 free-run 有效吗？ | 没有生成（2C 没有运行） | 未评估 |
| 是否具备进入 Stage 3 的条件？ | 2B 坍缩，Teacher 方法语义弱，Bypass 没有得到检验 | **NO** |

## 3. 根因（区分证据和推测）

**已有证据：**
1. **直接阻塞点是 2B 的 KD 优化，不是 Teacher 不可学。** 只读诊断（`q2z_probe.json`，不训练 Student）：
   - 用 projector 实际看到的那个冻结 Stage-1 h_seed 做闭式 ridge 回归，λ 只在训练集内部划分上选。
   - 48 道未见题上 KD = **0.433**，均值基线是 0.555：增益 0.122 [0.107, 0.137]，96% 的题为正；correct 比 shuffled 低 0.29 [0.26, 0.32]，98% 为正。
   - 可解释约 22% 的方差；用 mean-pool 特征时约 27%，KD 0.406。
   - 也就是说，Q→Z_T 中有可预测的、题目相关的部分，而且信息已经在 Projector 的输入里。端到端的 LoRA + 新 projector 在 40 步（lr 1e-4，clip 1.0）里只学到了公共方向，题目方差反而从 0.31 降到 0.024。
2. **Teacher 方法语义确实偏弱**（3/8 清楚切换；方法 token 上 Δ 只有 0.04）。即使 2B 学好，KD 的上限也有限：ridge 只解释 22–27% 的方差，同题 A/B 两条 Plan 之间的 Teacher KD 是 0.56，跨题是 1.12。
3. **Executor 对 latent 的依赖本身就很弱**：外部 Z_T 在未训练的 Executor 上 0/8 能控制方法；Exec CE 对各种 latent 干预的差别 ≤0.003 nat。
4. **Method Choice 信号太弱不是根因**：这个设置下 Choice 梯度的量级与主 loss 相当甚至更大。

**推测（未验证）：**
- 2B 坍缩的机制是：MSE 下降最快的方向是所有题共享的 Teacher 均值（Teacher 跨题 cos 0.465），而 projector 输出又被 RMS 归一化，于是优化先把所有 Z_S 拉向同一方向。题目相关的回归部分需要更多步或更好的初始化。
- 即使 2B 学好，自主 Z_S 也只能携带 Q 本身已有的信息，因为它是确定性的 f(Q)。能否打破 Bypass 只能由 2C 的反事实 Choice 加干预评估来检验，这一点本轮没有证据。

## 4. 回答四个问题

1. **根因更可能是哪一个？** 本轮直接卡在 **KD 目标坍缩**（2B 优化只学到了平均向量），有直接证据。Teacher 2A 语义偏弱是次要的上限问题，也有证据。“Method Choice 信号太弱”在这个设置下**不成立**，梯度审计可以证明。“Executor 忽略自主 Z”在 2B 阶段可以观察到，但这是 Z_S 坍缩的必然结果，不能单独归因；它需要 2C 才能检验。
2. **2B KD-only 学到了什么？** 只学到了 Teacher 的公共方向（Z_S 与 Teacher 的平均 cos 从 0 升到 0.42），没有学到题目相关信息：correct 等于 shuffled，reader 也读不出差别。**2C 两支没有运行，没有 Choice 与 Control 的对比数据。**
3. **是否已经拥有“Question → 自主 latent → Explicit CoT + Answer”且 latent 起作用的模型？** **没有。** 管线可以跑通，梯度和接口都验证过，但阻塞在 2B：自主 latent 在各题之间几乎相同，对执行没有可测的影响。
4. **下一步最少改哪里？** **只改 2B 的初始化/优化，让 Z_S 先变成题目相关的。** 具体做法：用训练集上的闭式 ridge 解（h_seed→Z_T，本轮已证明在未见题上有效）初始化 projector 的 Q→z1 映射，或者先只训练 projector 做 KD；以“未见题 KD < 均值基线 0.555，且 correct < shuffled 的 CI 不跨 0”作为进入 2C 的门槛。2C 的设计与预算保持不变。

## 5. 文件
- 文档：`ASSET_MANIFEST.md`、`DATA_AUDIT.md`、`TEACHER_SEMANTIC_AUDIT.md`、`ENGINEERING_GATES.md`、`RUN_CONFIG.yaml`、`REPRODUCE.md`、`STATUS.md`。
- 指标：
  - `train_metrics.csv`（Tiny + 2B 逐步）；
  - `stage2b_kd_results.csv`（step 0/20/40 + ridge/1-NN 探针）；
  - `choice_gradient_audit.csv/json`；
  - `autonomous_latent_interventions.csv`（27 对 × 3 个 checkpoint）；
  - `stage2c_*_results.csv`（NOT_RUN）。
- 评估原始数据：`evaluations/summary_2b_step{0,20,40}.json`、`rows_*.jsonl`；`teacher_reader_freerun.json`；`q2z_probe.json`。
- Checkpoint（只包含本轮新产生的）：`checkpoints/2b_step{0,10,20,40}.pt`。
- 局限：held-out 只有 8 对，Choice 有效对只有训练 8 + held-out 5；训练对的 Teacher 方法判定只有自动判定；未见题评估用了 val 的前 48 条。
