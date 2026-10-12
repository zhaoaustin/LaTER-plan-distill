# Stochastic Latent Planning Pilot（z1 加噪 + Set Matching + Method Choice）最终报告

**结论：`FAIL_COLLAPSE`（Gate 3 Tiny Overfit 判定为 `TRAIN_FIT_FAIL`；工程部分 `ENGINEERING_PASS`）。**
- **工程 Gate 全部通过**：噪声确实沿 recurrence 从 z1 传到 z8；SetKD 满足置换不变；三项 loss 回传到 Projector 和 LoRA 的梯度都非零。
- **坍缩**：训练 25 步后，Student 把噪声压掉了。同题两个 ε 的 Z 距离从 0.21 降到 0.003，z8 cos 从 0.906 升到 0.9999，而 Teacher A/B 之间的距离约为 0.6。两个样本一起往 A/B 的中点靠。
- **训练题上没有方法条件效应**：配对 Δ 两边都为正的只有 5/19（step0 为 2/19）；ε 之间偏好符号发生翻转的题数为 0/19；Choice loss 一直停在 ln2。
- **held-out 上同样没有**：Δ 两边都为正 2/8；偏好符号翻转 0/8；自由生成中按 ε 切换方法 0/8；不同且正确的方法数@8 最多为 1。
- **Latent bypass 没有解决**：self / 交换 / σ=0 / zero / mean / cross-question 六种条件下的 Execution CE 相差不超过 0.001。
- **按 Gate 3 规则停训**：120 步预算只用了 28 步（3 步 smoke + 25 步 Tiny Overfit），没有进入主 Pilot。
- **下一步只建议一件事：修 stochastic generation**，原因见第 7 节。

## 1. 运行概况

| 项目 | 数值 |
|---|---|
| optimizer steps | **28/120**：3 步 Gate smoke + 25 步 Tiny Overfit；主 Pilot 没有运行 |
| 题目数 | 训练用 19 道不同的题（Tiny 阶段只用其中 12 道）；held-out 8 道；Choice 可用的：训练 8 道、held-out 5 道 |
| GPU | Slurm 48907334 / e4-1，2×NVIDIA B200。训练用 torchrun world=2，手动 all-reduce，step 1 和 step 25 两卡参数完全一致；评估两卡各跑一个 shard |
| 时间 | 训练 38 s（1.5 s/step，每步 4 道题 × K=2）；Gate 约 3 min；评估约 75 min（含 27 题 × 14 条件 × 最多 2048 token 的 greedy 生成） |
| 显存峰值 | 训练 rank0 40 GB；Gate 单卡 50 GB |
| 历史结果复用 | 2A 的 Z 缓存和 step300、2B step-50 adapter（只作为冻结的 Teacher 映射）、2B 的 held-out label audit、2A 的自由生成审计分层 |

## 2. 实际管线与防泄漏（代码：`later/scripts/stage2s/stochastic_pilot.py`）

```
Student（Stage-1 Qwen3-4B + 一套共享 LoRA r16 q/v/o + LaTER LatentProjector，新初始化，输出 RMS = token RMS）
  chat(Q) + '<EXPLORATION>\n' ─► h_seed ─► z1 = proj(h_seed) + 0.10·0.02195·ε        [Student.latents]
  z_{t+1} = proj(h(z_1..z_t))，KV-cached recurrence，带梯度，LoRA 生效，送回的是加噪后的 z1
  Executor：chat(Q) + '<EXPLORATION>\n' + [z1..z8] + '</EXPLORATION><EXECUTION>\n' + targets
           一次完整 forward，use_cache=False，不复用 Planner 的 KV                     [Student.score]
Teacher（只在训练时使用，冻结）：T(P) = Adapter2B_step50(Compressor2A_step300(P))，写入同一组 8 个 input-embedding slot
```
- **防泄漏**：
  - Student 的 prompt 只由 `render_prompt(Q)` 加 `plan_open_ids` 组成；Gate 检查过其中不含 P_A、P_B、E_A、E_B 的文本。
  - P 只送进 `Student.teacher()`，该函数从 2A 缓存取 Z，再经过 2B adapter。
  - E 只出现在 CE、Choice 的 target 里。
  - Choice 的前缀 C 是中性的，例如 "1. Use the"、"1."，见 METHOD_SPAN_AUDIT.md。
- **SetKD 的匹配方式**：每道题单独匹配；匹配结果同时决定哪个 Z 接 E_A/D_A，Exec 和 Choice 用的是同一套匹配。

## 3. 训练曲线（metrics.jsonl，Tiny 阶段 12 对，每步 4 道题）

| step | L_set | L_choice | L_exec | 同题两样本距离 d(Z1,Z2) | z8 cos(Z1,Z2) | Δ 两边都为正 | 梯度范数（Projector / LoRA） |
|---|---|---|---|---|---|---|---|
| 1 | 2.007 | 0.743 | 0.299 | 0.210 | 0.906 | 1/4 | 332 / 50 |
| 5 | 1.915 | 0.700 | 0.248 | 0.049 | 0.949 | 0/4 | 10.7 / 1.5 |
| 10 | 1.576 | 0.740 | 0.236 | 0.010 | 0.995 | 0/4 | 2.6 / 0.43 |
| 15 | 1.277 | 0.692 | 0.210 | 0.005 | 0.9994 | 2/4 | 0.92 / 0.19 |
| 20 | 1.045 | 0.712 | 0.210 | 0.003 | 0.9999 | 2/4 | 0.53 / 0.14 |
| 25 | 1.018 | 0.698 | 0.163 | 0.003 | 0.9999 | 0/4 | 0.48 / 0.19 |

- **各项变化**：
  - Teacher 同题 A/B 的距离一直在 0.54–0.67；
  - SetKD 在下降，但两个样本是一起往中点走；
  - Exec CE 在下降，这是 LoRA 带来的通用执行改进，与 Z 无关（见第 5 节）；
  - Choice 一直约为 0.69 = ln2。
- **匹配方向**：匹配到 1A2B 的比例在 0.25–0.75 之间随机波动，说明 ε 和 A/B 之间没有形成稳定的对应关系。
- **step 1 的梯度诊断**：Projector 梯度中，set 153、exec 64、choice 9.2；乘上 λ=0.1 后 Choice 的有效梯度只有 0.92，约为 SetKD 的 1/170，基本被 SetKD 盖住。按任务书规定没有改系数。
- **step 25 的 4 个训练例子**（匹配 / Δ_A / Δ_B）：
  - 00645：1B2A / −0.004 / −0.017
  - 00653：1B2A / +0.003 / −0.003
  - 00626：1A2B / −0.004 / +0.004
  - 00344：1A2B / +0.004 / −0.002

## 4. 最重要的对比表（`comparisons_step*.csv`，after_tiny_overfit = step 25）

Teacher-forced 指标（方法偏好 pref = s(D_A|Z) − s(D_B|Z)，按 token 平均，K=8 个固定评估 ε）：

| 指标 | 训练 19 题：step0 | 训练 19 题：step25 | held-out 8 题：step0 | held-out 8 题：step25 |
|---|---|---|---|---|
| 配对 Δ 两边都为正（ε0/ε1 按 Teacher 距离匹配） | 2/19 | 5/19 | 1/8 | 2/8 |
| 配对 Δ 均值 [按题 95% CI] | −0.0021 [−0.0046, 0.0] | +0.0004 [−0.0014, 0.0021] | −0.0013 [−0.0055, 0.0035] | +0.0021 [−0.0015, 0.0049] |
| **8 个 ε 中出现偏好符号翻转的题数** | 0/19 | **0/19** | 0/8 | **0/8** |
| pref 在 ε 之间的标准差 | 0.0082 | 0.0054 | 0.0088 | 0.0060 |
| \|pref(self 均值) − pref(σ=0)\| | 0.0066 | 0.0050 | 0.0089 | 0.0054 |
| \|pref(self) − pref(zero / mean / crossq)\| | 0.025 / 0.009 / 0.011 | 0.019 / 0.006 / 0.006 | 0.019 / 0.012 / 0.007 | 0.023 / 0.005 / 0.005 |
| 同题 8 个 Z 的平均 cos | 0.915 | **0.9985** | 0.930 | **0.9984** |
| 同题 Z 到 σ=0 Z 的距离 | 0.152 | 0.0016 | 0.114 | 0.0016 |
| 8 个 ε 里同时出现“更近 T_A”和“更近 T_B”的题数 | 3/19 | 0/19 | 0/8 | 0/8 |
| Exec CE：self 匹配 / self 交换 / σ=0 | 0.2546 / 0.2550 / 0.2550 | 0.1788 / 0.1788 / 0.1788 | 0.2257 / 0.2296 / 0.2279 | 0.1716 / 0.1718 / 0.1719 |
| Exec CE：zero / mean / crossq | 0.2379 / 0.2603 / 0.2614 | 0.1799 / 0.1788 / 0.1789 | 0.2154 / 0.2266 / 0.2305 | 0.1718 / 0.1721 / 0.1721 |
| **外部 Teacher（诊断用，单独列出）**：pref(T_A) − pref(T_B) 均值，方向正确的题数 | +0.056，19/19 | +0.048，18/19 | +0.0005，4/8 | −0.004，2/8 |
| Exec CE：Teacher | 0.2336 | 0.1779 | 0.2125 | 0.1717 |

自由生成（step25，greedy，最多 2048 token，每题 14 个条件；方法由人工判定）：

| 指标 | 训练 19 题 | held-out 8 题 |
|---|---|---|
| ε0 / σ=0 / zero / mean / crossq 的答对题数 | 16 / 15 / 12 / 14 / 14 | 5 / 4 / 5 / 4 / 5 |
| 8 个 ε 的 Pass@1 均值，Pass@8 | 0.74，16/19 | 0.61，6/8 |
| 正常结束的比例（ε） | 0.87 | 0.95 |
| **人工判定：同题 ε 之间换了数学方法** | **2/19**（00653：因式分解 vs 试值；00422：通项公式 vs 同余），但 σ=0、zero、mean、Teacher 也会随机落到两边，见下文 | **0/8** |
| **不同且正确的方法数@8** | 00653 和 00422 各为 2，其余题 ≤1 | 5 题为 1，3 题为 0（00574、00309、00486 全错） |
| 外部 Teacher A/B 自由生成的方法切换（诊断用） | 人工判定 0 题（自动指标标出的 3 题都是措辞差异；00653 的 teacher_A 反而走了 B 方法） | 0/8（00336、00569、00479、00305、00507、00309、00486、00574 两边方法都相同） |

### 正反例（held-out，step25；全文见 `evaluations/generations_step25_tiny.md`，逐条记录见 `heldout_predictions_step25_tiny.jsonl`）
- **00336**（A = Lagrange，B = Cauchy-Schwarz）：8 个 ε、σ=0、zero、mean、T_A、T_B，**13 条生成全部走 Cauchy-Schwarz**，逐字相同；只有 crossq（换成别的题的 Z）走了 Lagrange。说明 Z 有能力扰动方法，但这种扰动没有受到控制。
- **00305**（A = 代入 y=6/x，B = 因式分解）：所有条件都走代入法，只是措辞分成两组。
- **00569**：所有条件都是“改写为 3 + 1/(x²+x+1)”，只有措辞差异。
- **00574**：所有条件都是“改写成 2 和 3 的幂，再试小值”，全部答错。
- **00309、00486**：14 条生成逐字相同，全部答错。
- **训练题 00653**（反例：看起来像多样性，其实不受控）：ε1/3/5/6 走平方差分解，ε0/2/4/7 走试值；但 σ=0、zero、mean 也走分解，T_A 却走了试值（与 Plan A 相反）。这是 greedy 解码在临界点附近的混沌敏感性，不是计划控制。
- **训练题 00645 / 00496 / 00535**：所有 ε 的方法都相同，只是措辞不同。

## 5. 解释：我们是否只是把噪声变成了不同的向量？

**连“不同的向量”都没有保住。**
- **step0**：随机初始化的 Projector 会把 z1 上 10% 的噪声放大，同题 8 个 Z 的 cos 约为 0.92。
- **训练后**：Student 主动把对 ε 的敏感度压低了约 70 倍（到 σ=0 Z 的距离 0.15 → 0.0016）。
- **原因**：每一步 ε 都重新采样，方向本身没有固定含义。在 min 匹配下，降低方差、把两个样本都放到 T_A 和 T_B 之间，是期望 loss 下降最快的方向。
- **Choice 帮不上忙**：要把噪声空间稳定地切成“A 区”和“B 区”，需要 Choice 提供足够强的信号，而它的有效梯度只有 SetKD 的 1/170。

**Executor 依赖**：六种条件的 Exec CE 相差不超过 0.001，与 Stage 2B 一样，执行段没有使用 Z。CE 从 0.25 降到 0.18 对所有条件都一样，属于 LoRA 的通用改进。

**Teacher 信号的局限**：外部 T_A/T_B 只在训练题上让 pref 往正确方向偏一点（step0 时 19/19，+0.056 nats）。但 step0 的 LoRA 还没训练，这个效应只能来自 2B adapter：它在 2B 中正是用这 20 个训练对训练过的，属于间接泄漏。在 held-out 上只有 4/8 和 2/8，与随机相当。这说明问题不只在 Student 生成，**外部 Z 本身在未见题上也几乎不改变 Executor 的方法选择**。

**四个问题**：
- **Q1 不同 ε 能否生成不同的 Z？** 结构上可以（Gate 中 z8 的相对差为 0.53），但训练把这种差异消掉了。
- **Q2 Z₁/Z₂ 的差异是否对应 Plan A/B？** 不对应。8 个 ε 里同时更接近 T_A 和 T_B 的题数为 0。
- **Q3 自主 Z 能否控制 Explicit Execution？** 不能。各条件 CE 一样，方法偏好不随条件变化。
- **Q4 训练题和未见题分别如何？** 两边都没有信号。训练题都学不到，就无法区分“泛化不足”和“机制本身不成立”。本轮的主因在 stochastic generation 一侧（坍缩），Executor 一侧的 bypass 仍然存在。

## 6. 局限
- 只跑了 25 个训练步，但已经满足 Gate 3 的 STOP 条件：训练题上没有可证实的效应，而且坍缩趋势单调，不像是训练不足造成的。
- 训练题只有 19 道，Choice 可用的只有 8 道；held-out 只有 8 道，无法得出任何 `SCIENTIFIC_PASS` 级别的结论。
- Teacher 映射用的是 2B adapter，它在 2B 中没有被证明能驱动 Executor。
- 方法判定由人工完成，只读了开头和关键词。

## 7. 下一步（只给一条）
**修 stochastic generation**，也就是让 ε 能以稳定的方式对应到不同的方法区域。目前高斯噪声加 min 匹配在实际训练中会退化成“平均两种 Plan”。最小的改法有两种，都需要你批准后才做：
- 把连续 ε 换成一个离散的 K 选 1 规划码，例如 2 个可学习的 seed embedding，再用 winner-take-all 的 set matching；
- 或者给 Choice/SetKD 加上同题样本之间的显式分离项。

在这一步能让训练题上出现稳定的 A/B 分流之前，不建议扩大数据，也不建议进入 Stage 3。

`STAGE3_RL_DESIGN_RISK`：`z1 = μ + σε` 经过确定性 recurrence 后，整条 Z 轨迹的 log-prob 只取决于 ε 的高斯密度，与 recurrence 无关，不能直接套用 token 级 GRPO 的 ratio。本轮没有涉及 RL。

## 8. 复现
```bash
cd /scratch/hz33227/e2c/stage2/LaTER   # 分支 exp/stochastic-latent-setkd
M=/scratch/hz33227/e2c/Explore-Execute-Chain/outputs/e2c_qwen3_4b_sft/best_hf
B=/scratch/hz33227/e2c/stage2/outputs/stage2b_executor_50step
O=/scratch/hz33227/e2c/stage2/outputs/stochastic_latent_setkd_pilot_20261012
export PYTHONPATH=$PWD; PY=~/.conda/envs/later-b200/bin/python
srun --jobid=48907334 --overlap --gres=gpu:1 $PY -m later.scripts.stage2s.stochastic_pilot --mode gates --model $M --s2b $B --out $O
srun --jobid=48907334 --overlap --gres=gpu:2 ~/.conda/envs/later-b200/bin/torchrun --nproc_per_node 2 -m later.scripts.stage2s.stochastic_pilot \
     --mode train --model $M --s2b $B --out $O --tiny_steps 25 --stop_at 25
for s in 0 1; do srun --jobid=48907334 --overlap --gres=gpu:1 $PY -m later.scripts.stage2s.stochastic_pilot --mode eval --model $M --s2b $B --out $O \
     --ckpt $O/checkpoints/after_tiny_overfit.pt --shard $s/2 --gen --tag _tiny & done; wait
$PY -m later.scripts.stage2s.aggregate $O 25_tiny
```
- **随机种子**：数据顺序 42，Projector 初始化 0；ε 由 sha1(元组) 作种子生成，训练时的元组为 ("train", step, qid)，评估时为 ("eval", qid)。
- **checkpoint**：`checkpoints/step0.pt`、`checkpoints/after_tiny_overfit.pt`（= step25.pt）；sha256 见 ASSET_MANIFEST.md。
- **未 push，没有覆盖任何旧 checkpoint。**
