# ENGINEERING GATES（Gate 1，`--mode gates`，单卡）

**14/14 硬性单测通过（ENGINEERING_PASS）。** 原始输出见 `gates.json`。

| 检查 | 结果 | 证据 |
|---|---|---|
| 1 共享参数 | PASS | 只有 1 套 LoRA（['default']，['o_proj', 'q_proj', 'v_proj']，9.73M）加 projector 13.11M；自主分支和 Choice 分支调用同一个 peft 模型对象 |
| 2 Plan 只给 Teacher | PASS | Student prompt 解码后不含 Plan、Execution 片段或方法标签；prompt 尾部为 `\n<|im_start|>assistant\n<EXPLORATION>\n [8 slots] </EXPLORATION><EXECUTION>\n` |
| 2 Teacher 停梯度 | PASS | Z_T requires_grad=False |
| 3 Latent 自主 | PASS | 把 Teacher 缓存改成常数后，Z_S 逐位相同 |
| 3 确定性 | PASS | σ=0，重复两次完全相同 |
| 4 8 个槽都可见 | PASS | 逐个把 slot 置零后，第一个 Execution token 的 logits 变化：5.62, 4.00, 3.04, 2.59, 1.99, 3.54, 1.41, 1.84 |
| 4 训练与生成路径一致 | PASS | teacher-forcing 路径与 KV-cache 生成路径的首 token argmax 相同；每个条件都重新计算完整 forward（use_cache=False），不复用其他条件的 KV |
| 5 梯度路径 | PASS | KD/Exec/Choice_self 都在 projector、首层、LoRA 和 dz1 上有非零有限梯度；Choice_ext 只到 LoRA（projector=0，符合预期）。范数：exec proj 34.4 lora 4.63 dz1 0.253；choice_self proj 36.4 lora 5.09；choice_ext lora 0.168 |
| 6 Loss mask | PASS | target 以 ` $$</EXECUTION><|im_end|>` 结尾，label 只覆盖 E+A |
| 6 Span 一致 | PASS | 短 forward 与全长 forward 在 D_A 上的得分相同：[-0.5508396625518799, -0.5508396625518799]；C=`1. Use the` |
| 7 4 格检查 | PASS | 两个 Z 设成相同时 m_TA=m_TB=-0.1260，不会出现两个方向同时成功 |
| 8 干预有限 | PASS | self/zero/mean/crossq/teacher_B/no-latent 的 span 得分都是有限值：self -0.551, zero -0.579, mean -0.536, crossq -0.543, teacher_B -0.575, nolatent -0.593 |
| 10 Reload | PASS | 保存并恢复 LoRA、projector、optimizer、RNG 后 loss 相同 [2.0260064601898193, 2.0260064601898193] |
| 10 主干冻结 | PASS | base 权重逐位不变 |

## 9 Tiny Overfit（10 个 optimizer step，计入预算；4 个单样本 + 2 个 Choice 有效对，三项 loss，λ_choice=1 为临时值）

- KD correct：[1.988, 1.96, 2.008, 2.038] → [1.855, 1.862, 1.914, 1.935]；shuffled：[1.969, 2.004, 2.052, 1.994] → [1.871, 1.905, 1.95, 1.865]。correct<shuffled 的样本从 2/4 变为 3/4。
- Choice（两个对的默认 Plan 都是 B）：m_TA−m_TB 从 0.017 / -0.026 变为 0.053 / 0.072，两对都转向正确方向；Choice loss 从 2.007 降到 1.864。
- Exec CE 在 0.339–0.355 之间，没有下降；第 4 步出现一次梯度尖峰（裁剪前 1.2e4），被 clip 到 1.0。
- 判定：**弱通过**。方向都对，但 KD 下降很慢（2.00→1.89）。这已经预示了 2B 在 40 步内可能拟合不足。
