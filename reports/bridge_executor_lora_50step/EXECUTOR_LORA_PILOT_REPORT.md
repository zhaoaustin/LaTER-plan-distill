# Executor LoRA Conditioning 50-Step Pilot — 报告

| 指标 | 原 B-50 Bridge + 冻结 Executor（Step 0） | B Bridge 冻结 + Executor LoRA（Step 50） |
|---:|---:|---:|
| held-out 平均配对 margin（前 64 token，对同题交换 Z）[95% CI] | −0.0021 [−0.014, +0.010] | +0.0141 [−0.0067, +0.036] |
| held-out margin 为正的比例（16 个） | 6/16（0.38） | 9/16（0.56） |
| 两个方向都为正的题数 | 0/8 | 3/8 |
| Greedy 正确切换 / 8（人工判定） | 1/8（s1val-00305） | **0/8**（另有 1 题方向相反的切换：s1val-00336） |
| 答案正确 / 16 | 8 | 11 |
| 正确 Z 比别题 Z / 均值 Z 好多少（held-out） | +0.0026 / **−0.0096** | +0.030 [−0.019, +0.085] / **−0.018** |
| 训练题配对 margin（对交换 Z） | +0.020（85% 为正） | **+0.274（95% 为正）** |
| held-out 整段 CE（正确 Z） | 0.211 | 0.238（变差） |
| LoRA 可训练参数 | 0 | 9,732,096（约 0.24%） |
| 训练用时 / 显存峰值 | — | 81 s / 74.5 GB |

- 整段 margin（对交换 Z）：step 0 为 −0.0006，step 50 为 +0.0002。
- step 25 的中间结果：held-out margin +0.0026（对交换 Z）、−0.0063（对均值 Z）；train margin +0.027。

## 设置（相对 A/B/C/D 中的 B 组，只多了一个变量：Executor LoRA）

- **Bridge**：用 B 组 step 50 的 checkpoint 和训练集标准化统计量，完全冻结（逐位核验）。
- **Z 的来源**：所有 Z 只用冻结、不带 LoRA 的 Stage-1 计算一次并缓存。接入 LoRA 后关闭 adapter 在线重算，与缓存的最大差异为 0，所以 Bridge 的输入从未经过 LoRA。
- **LoRA 配置**：r=16，α=32，dropout 0，作用于 q/v/o_proj。用的是任务书默认值；仓库里唯一的 LoRA 配置是给 14B SFT 用的（r=64、all-linear），没有沿用。
- **骨干**：训练前后逐位不变。
- **Loss**：与 B 组的实际实现完全相同，即整段 CE + 0.1·ranking，ranking 只在前 64 个执行 token 上计算。
- **其余设置**：数据、顺序（seed 1234）、A/B 随机互换、每步 4 对、AdamW（lr 1e-4，wd 0.01）、梯度裁剪 1.0，都与 B 组相同。训练 50 步，没有延长。
- **Step 0 复现了 B-50**：margin、切换题和正确数都逐项一致。

### Sanity

- 全部硬 Gate 通过，详见 `sanity_checks.md`。
- 第一次运行停在了“LoRA 初始化后评分是否不变”这一项。原因是测试拿两种不同的批次组成做比较，属于测试写错，并非 LoRA 改变了输出：B=0 时的实际差异只有 6e-8，与重复运行的噪声相当。修正后差异为 1.5e-8。第一次运行没有进行任何训练。

## 结论

- **训练题上 LoRA 很有效**：margin 从 0.02 升到 0.27。
- **held-out 上没有泛化**：
  - 对同题交换 Z 的 margin，CI 包含 0；
  - **正确 Z 不如均值 Z 和全零 Z**：分别为 −0.018 和 −0.019；
  - held-out CE 变差；
  - 正确切换 0/8，step 0 时原有的 1 题切换也丢了。
- **值得注意的现象**：s1val-00336 两种方法互换了，Z_A 产生 Cauchy-Schwarz，Z_B 产生 Lagrange。可见 LoRA 之后 Executor 对 Z 更敏感，Z 能影响它选哪种方法；但在没见过的题上，Z 与方法的对应是错的。
- 这一模式是在 20 道训练题上记住了 Z 与解法的对应，属于过拟合，而不是学会了读出 Plan 信息。

## 三句话回答

1. **Executor LoRA 没有让未见过的题的执行更依赖正确的 latent（否）。** held-out 上对交换 Z 的 margin 为 +0.014 [−0.007, +0.036]，正确 Z 反而比均值 Z 差 0.018；正确切换从 1/8 变成 0/8；训练题 margin 却从 0.02 升到 0.27。
2. **主要表现是没有泛化，而不是 latent 被忽略。** 训练后 Executor 对 Z 明显更敏感：训练题 margin 大幅上升，held-out 也出现一次方向相反的切换。但它学到的是这 20 道训练题上 Z 与解法的对应，没有学到能迁移的 Plan 信息。8 道题不足以进一步区分具体原因。
3. **现在不值得做 Bridge + LoRA 联合训练。** 冻结 Bridge 时，单独加 LoRA 已经只会在训练题上过拟合；联合训练会增加可拟合的参数，在 20 对数据上只会让记忆更严重。应先扩大双 Plan 数据（例如 200 对以上，流程已跑通），或改用更密集的监督（如以文字 Plan 条件下 Executor 的状态为蒸馏目标），再考虑联合训练。

## 文件

- `run_manifest.json`
- `sanity_checks.md` / `sanity_checks.json`
- `post_training_checks.json`
- `train_log.jsonl`
- `heldout_step{0,25,50}.json`
- `per_question_pair_metrics.jsonl`
- `heldout_generations.md`（含人工判定）
- `generations_step50.jsonl`
- `cached_Z.pt`
- `lora_adapter_step50/`
- `code_patch.diff`
- `run.log`、`run_attempt1_gate_test_bug.log`
