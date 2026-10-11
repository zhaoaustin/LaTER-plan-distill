# SANITY GATES（全部 PASS）

| Gate | 证据 |
|---|---|
| 1 checkpoint 身份 | Stage-2A step300 sha256 7c63e24d5dd4a953…；重新计算的 held-out 配对 Δ 为 0.109789，与报告值的差为 0；Stage-1 是 best_hf（step 800） |
| 2 无泄漏 | 前缀尾部为 '><|endoftext|></EXPLORATION><EXECUTION>\n'；前缀中没有 Plan、Execution 或答案文本；encoder 只读 plan_norm；训练和验证样本已过滤 Plan 中含答案的情况（135 + 13 条） |
| 3 slot 与形状 | Z 为 [8, 2560]，等于 hidden 2560；slot 内容等于 Adapter 输出（误差 0.0）；position 连续；4 条序列的前缀一致；logits 有限；没有使用 0.6B 的 Adapter |
| 4 首 token | 首 token CE 手算 0.581834，评分函数 0.581834 |
| 5 反事实注入 | 交换 Z 后 slot 和 logits 都改变；零 Z 走同一条 Adapter 路径且输出有限；每次前向都是 use_cache=False；训练路径与生成路径首 token 的 argmax 一致 |
| 6 冻结与梯度 | optimizer 中只有 Adapter 和 LoRA；骨干逐位不变（gate 后核验；训练后参数和也不变）；Adapter 和 LoRA-B 梯度有限且非零；骨干梯度为 None |
| 7 双卡 | torchrun world=2；每张卡 4 个单样本 + 1 对 Plan，数据分片不同（step 1 两卡本地梯度范数 [0.197, 1.787]）；梯度 all-reduce 取平均；step 1 和 step 50 两卡参数完全相同；显存峰值 [66.2, 74.6] GB；step 10 时利用率 98% / 94% |
| 8 存取 | 重新加载后 loss 分别为 0.3072878 和 0.3072877（bf16 容差内） |
| 9 小样本过拟合 | 6 条样本训练 5 步，CE 0.287 → 0.255 |

- 第一次运行时 Gate 3 的“slot = Adapter 输出”失败，误差 0.00049。原因是测试在 fp32 下计算 Adapter，而实际打分在 bf16 autocast 下计算，属于测试写错。改成相同精度后误差为 0。记录见 sanity_gates_single_gpu_attempt1.json。
