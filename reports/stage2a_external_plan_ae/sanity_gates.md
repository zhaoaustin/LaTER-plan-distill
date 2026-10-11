# Sanity Gates（全部 PASS）

| Gate | 证据 | 结果 |
|---|---|---|
| 可训练参数 | compressor 104,944,640 / adapter 2,622,464 / decoder LoRA 10,092,544；Decoder 非 LoRA 可训练参数 0 | PASS |
| 1 不泄漏 Question | train/val 文件不含 question 字段；所有样本的 Decoder 前缀都等于同一句固定指令；8 个 soft slot 写入误差为 0 | PASS |
| 2 不泄漏未来 token | 改动目标后半段后，前半段 logits 最大差 0.0；后半段确有变化 | PASS |
| 3 只依赖 latent | 交换 Z 后 logits 改变；generate 首步与 teacher forcing 首个位置的 logits 最大差 0.0，argmax 一致 | PASS |
| 4 冻结 | Decoder 主体逐位不变（gate 后与训练后都核验过）；optimizer 中只有 compressor、adapter 和 LoRA；4B 在训练进程中从未加载，缓存前后不变 | PASS |
| 5 梯度 | ‖∂L/∂Z‖ = 2.896；compressor、adapter 和 LoRA-B 的梯度有限且非零；Decoder 主体梯度为 None | PASS |
| 6 首 token / 移位 / padding / EOS | 手算首 token CE 与评分函数一致（8.2024）；目标长度 = 文本 token 数 + EOS；padding 位置不计入 | PASS |
| 7 小样本过拟合 | 16 条样本训练 40 步：loss 1.670→0.0008；打乱 Z 比正确 Z 的 CE 高 0.200，16/16 为正（预注册标准：>0.1 且 ≥14/16） | PASS |
| 8 存取 | 重新加载后 loss 完全相同；下一步更新最大差 0.0 | PASS |

- 训练后核验：Decoder 主体逐位不变（True）；训练 138 秒，显存峰值 13.1 GB。
