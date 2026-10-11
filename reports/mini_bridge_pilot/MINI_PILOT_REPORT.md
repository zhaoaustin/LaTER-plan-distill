# Mini Bridge Pilot — 同题双 Plan 的 Latent-conditioned Executor（报告）

**结论：`NO_SIGNAL_IN_MINI_PILOT`。**
- 训练 50 步后，冻结的 Executor 不会因为 Z_A / Z_B 不同而切换解题方法。
- Bridge 学到的是与 Plan 无关的通用 latent：Z_A 与 Z_B 的 cos = 0.9988，CE 下降对 correct 和 swapped 相同。
- 只看到极小的 train margin（0.0006）；held-out margin 的 95% CI 包含 0。
- held-out 没有正向趋势，按规则没有延长到 100 步。
- 这只是 28 道题的可行性实验，不能排除更大的 Bridge、更强的 ranking 权重或 LoRA 的可能性。

## 设置

- **Executor / 编码器**：Stage-1 E²C Qwen3-4B step 800
  - 路径：`/scratch/hz33227/e2c/Explore-Execute-Chain/outputs/e2c_qwen3_4b_sft/best_hf`（= HF zhaohq/Qwen3-4B-E2C-SFT-only @ a6b0e2fe）
  - 全程 bf16、eval、`requires_grad=False`。
  - 不是 Stage-2 Student；没有加载或覆盖任何 Stage-2 checkpoint。
- **数据**：现有数据里没有同题双 Plan，因此临时生成了一份小数据。
  - 生成：用 vLLM 对 stage2_val 第 256–655 条（400 题）各采样 8 次，T=0.9，共 4 分钟（`data/samples_stage1_n8.jsonl`）。
  - 自动筛选：两个答案都经 math_verify 判对；方法词集合不同；各自的方法词在自己的 Execution 中出现得更多（plan-following 检查）。得到 89 个候选。
  - 人工审阅：全部 89 个候选都人工读过，接受 28 对，即 20 对 train 和 8 对 held-out，按题目隔离。
  - held-out 的方法标签见 `heldout_pairs.jsonl`，例如 Lagrange 乘子 vs Cauchy-Schwarz、判别式 vs 求导、Euler 定理 vs 阶为 3、因式分解 vs Vieta。
  - 有 3 个 Plan 的文本里含答案字符串，已记录。
- **Plan 编码器**：冻结的 Stage-1 读 `prompt(Q) + <EXPLORATION>\n + plan + \n`，取第 27 层 residual 在 plan token 上的状态。不含 Execution 和答案字段。
- **Bridge**（3937 万参数，唯一可训练的模块）：8 个 query 的 cross-attention，加残差 MLP，输出按 token-embedding RMS（0.02195）归一化。
- **Executor 输入**：`prompt(Q) + <EXPLORATION>\n + [8 slots] + </EXPLORATION><EXECUTION>\n + execution targets`。A/B 的前缀完全相同，不含文字 Plan。
- **Loss**：L = ½[CE(E_A|Z_A) + CE(E_B|Z_B)] + 0.1·[softplus(−m_A) + softplus(−m_B)]，s 为监督 token 平均 log-prob。
- **训练**：AdamW，lr 1e-4，每步 4 对，50 步；A/B 标号随机对换；seed 固定。

## 单测硬门槛（全部通过，`unit_tests.json`）

| 检查 | 结果 |
|---|---|
| Bridge 梯度 | 有限，norm 0.199 |
| Executor 可训练参数 | 0 |
| Executor 梯度 | 全为 None |
| 1 次 optimizer step 后 Executor 权重 | bitwise 不变 |
| 编码器输入是否含 Execution | 不含 |
| Executor 前缀 | A/B 相同，且不含 Plan |
| Z 写入 8 个 slot | 误差 0 |
| 4 条序列的前缀 ids / position / mask | 一致 |
| checkpoint 存取 | Z 一致（误差 0） |

## 结果

| | step 0（随机 Bridge） | step 50 |
|---|---|---|
| held-out 平均 margin（每 token log-prob） | +0.00014 [−0.00027, +0.00049] | +0.00025 [−0.00017, +0.00068] |
| held-out margin > 0 的比例（16 个） | 0.69 | 0.62 |
| held-out 两个 margin 都为正的题数 | 3/8 | 4/8 |
| held-out CE correct / swapped（Z_A↔Z_B） | 0.2156 / 0.2157 | 0.1703 / 0.1705 |
| train 平均 margin（margin > 0 比例） | 0.00003 | 0.00062（0.75） |
| cos(Z_A, Z_B)，held-out | 0.959 | **0.9988** |
| **Oracle（文字 Plan 放进 prompt，同一 Stage-1）** | | **margin 0.167，16/16 为正** |

- **CE 下降与条件无关**：CE 从 0.216 降到 0.170，但 swapped 降得同样多。Bridge 学到的是对 Execution 普遍有帮助的前缀，而不是 Plan 条件。训练中 Z_A 与 Z_B 反而趋同（cos 0.959 → 0.9988）。
- **ranking loss** 从 1.3864 只降到约 1.385。0.1 的权重下，其梯度远小于 CE 的梯度。
- **Oracle 证明这些题的两种策略本身是可执行、可区分的**：文字 Plan 条件下的 margin 是 latent 条件下的约 700 倍。自由生成时方法完全跟着 Plan 走（8/8 题），答案 14/16 正确。

## 自由生成（greedy，`heldout_predictions.jsonl`）

- 8 题中有 2 题，Z_A 与 Z_B 的生成**逐字相同**。其余 6 题文本略有不同，但方法倾向相同，**没有一题随 Z 切换方法**。
  - 例：Lagrange vs Cauchy-Schwarz 那题，Z_A 和 Z_B 都走 Cauchy-Schwarz。
- latent 条件的答案正确率：Z_A 4/8，Z_B 4/8。Oracle 文字 Plan：14/16。
- **格式异常**：latent 条件下，Executor 常在 `<EXECUTION>` 之后先写出一段 “1. … 2. …” 形式的计划，再求解。说明冻结的 Stage-1 并不把这 8 个向量当作“已读过的 Plan”。

## 判定与下一步建议（未执行）

- 结论是 `NO_SIGNAL_IN_MINI_PILOT`。
  - train margin 有微弱上升（0.0006），但只有 oracle 的约 0.4%；held-out 的 CI 包含 0；方法没有切换。
  - 也算不上 `TRAIN_ONLY`：train 侧同样几乎没有信号。
- **主要瓶颈**：CE 项可以不依赖 Plan 就降低，ranking 项的权重太小，所以 Bridge 收敛到一个通用前缀。Executor 冻结时，8 个 slot 也可能不足以“写入”方法信息。
- 若要继续，可做以下小规模对照（仍属诊断）：
  - 只用 ranking / contrastive 项，或把权重从 0.1 提到 1–10；
  - 用 Oracle 的文字 Plan KV 做蒸馏目标；
  - 加更多 slot；
  - 对 Executor 开小 LoRA（需用户批准）。
- 本次未做任何 Executor LoRA、Stage-2 训练或 RL。

## 用时（实测）

| 阶段 | 用时 |
|---|---|
| 读代码与数据 | 约 4 分钟 |
| vLLM 采样 | 4 分钟（首次遇到 ninja/flashinfer 报错，改用 FLASH_ATTN 后端，损失约 3 分钟） |
| 人工审阅与数据构建 | 约 5 分钟 |
| 实现与单测 | 约 5 分钟 |
| 训练 50 步 | 1.6 分钟 |
| 最终评估（含 32 次生成） | 4.5 分钟 |
| **合计（11:20 → 约 11:45）** | **约 25–30 分钟** |

- 硬件：1× B200（Slurm 48907334）。

## 复现

```bash
cd /scratch/hz33227/e2c/stage2/LaTER; M=/scratch/hz33227/e2c/Explore-Execute-Chain/outputs/e2c_qwen3_4b_sft/best_hf; D=/scratch/hz33227/e2c/stage2/outputs/mini_bridge_pilot
srun --jobid=48907334 --overlap --gres=gpu:1 bash -c "export PATH=$HOME/.conda/envs/trl-b200/bin:\$PATH VLLM_ATTENTION_BACKEND=FLASH_ATTN; python later/scripts/bridge/sample_dual_plans.py --model $M --val /scratch/hz33227/e2c/stage2/data/stage2_val.parquet --out $D/data/samples_stage1_n8.jsonl"
# candidates_auto.json: screening snippet in code_patch.diff notes; accepted list is fixed in build_pairs.py
~/.conda/envs/later-b200/bin/python later/scripts/bridge/build_pairs.py $D/data/samples_stage1_n8.jsonl $D/data/candidates_auto.json $D
for m in test train; do srun --jobid=48907334 --overlap --gres=gpu:1 ~/.conda/envs/later-b200/bin/python -m later.scripts.bridge.mini_bridge --mode $m --steps 50 --model $M --out $D; done
srun --jobid=48907334 --overlap --gres=gpu:1 ~/.conda/envs/later-b200/bin/python -m later.scripts.bridge.mini_bridge --mode final --ckpt $D/bridge_checkpoint/step50.pt --model $M --out $D
```

## 文件

- `MINI_PILOT_REPORT.md`
- `train_pairs.jsonl`、`heldout_pairs.jsonl`
- `heldout_predictions.jsonl`
- `train_log.jsonl`
- `final_metrics.json`（含逐题 margin 与 oracle）
- `unit_tests.json`
- `bridge_checkpoint/`（step0.pt、step50.pt）
- `data/`
- `code_patch.diff`
