# P0 Residual Amplification — 报告

**结论：Case C。** 把 Projector 输出的去均值残差放大到 2 倍、4 倍后，正确题目的 latent 仍然不比别的题目的 latent 更有用（correct ≈ shuffled）。残差对 Executor 输出的影响落在 bf16 数值噪声底以内。这个简单的放大方法**没有解除 bypass**。

本实验是在已有 Stage 2 checkpoint 上做的离线、受控的 latent 干预。alpha > 1 的 latent 不是模型自己递归产生的。本实验不是训练，也不代表任何性能提升。

## 设置

- **模型**：原始三 Loss checkpoint-500，L=8。
- **评估集**：128 题（val 前 128 条）。
- **校准集**：另外 128 题（val 第 128–255 条），用来计算每步的 μ_t。
- **干预**：z^(α)_{i,t} = μ_t + α (z_{i,t} − μ_t)，α ∈ {1, 2, 4}。
  - **注入方式**：用 `latent_override` 把 8 个 slot 的向量直接写入学生序列，再做一次完整 forward（use_cache=False）。没有沿用 learned 路径的 KV cache，所以 attention 状态与注入的 embedding 一致。
  - **Shuffled 对照**：取固定 derangement（seed 0）下另一道题的 latent，经过同样的变换后整段 8 步放进当前题的同一组 slot。Question、位置、targets 和 masks 都不变。
- **CE 口径**：学生视图中全部执行侧 target（E + A + 闭合标签 + EOS）的逐题均值，再对题目取均值。
- **KL 方向**：KL(correct ‖ shuffled)，在前 16 个执行 target 上取均值。
- **Bootstrap**：按题目配对重采样 1000 次。

## α=1 自检（通过）

| 检查 | 结果 |
|---|---|
| override(learned Z) vs 原 builder 路径 | logits 最大差 0.0 |
| α=1 变换后的 latent vs 原 latent | 最大差 1.2e-10 |
| α=1 变换后 vs builder 路径 | logits 最大差 0.0，CE 差 0.0 |
| 与之前诊断的一致性 | swap_all 的 ΔCE 为 7.7e-5（只算非重复 token）；这次 α=1 的 gain 为 8.0e-5（算全部执行 token），两者一致 |

## 结果（128 题）

| 条件 | latent RMS / α1 | 去均值 RMS / α1 | 跨题 cos | CE correct | CE shuffled | **gain = CE_s − CE_c [95% CI]** | first16 gain [95% CI] | KL first16 |
|---|---|---|---|---|---|---|---|---|
| α=1 | 1.000 | 1.00 | 0.99901 | 0.246074 | 0.246154 | +8.0e-5 [−1.4e-5, +1.7e-4] | +5.0e-4 [−4.0e-4, +1.3e-3] | 2.05e-4 |
| α=2 raw | 1.0015 | 2.00 | 0.99605 | 0.246074 | 0.246028 | −4.5e-5 [−1.4e-4, +5.1e-5] | −4.7e-5 [−8.2e-4, +7.9e-4] | 2.01e-4 |
| α=2 RMS-matched | 1.000 | 2.00 | 0.99605 | 0.246100 | 0.246044 | −5.6e-5 [−1.7e-4, +7.3e-5] | +1.6e-5 [−9.5e-4, +9.7e-4] | 2.11e-4 |
| α=4 raw | 1.0076 | 4.00 | 0.98448 | 0.246058 | 0.246032 | −2.6e-5 [−1.3e-4, +7.8e-5] | +6.2e-4 [−2.5e-4, +1.5e-3] | 1.99e-4 |
| α=4 RMS-matched | 1.000 | 3.95 | 0.98448 | 0.246066 | 0.246118 | +5.3e-5 [−5.0e-5, +1.4e-4] | −6.8e-4 [−1.7e-3, +3.3e-4] | 2.11e-4 |

- **尺度**：raw 版本整体 RMS 最多只变了 0.76%，远低于 10% 的阈值，按规定其实不必做 RMS-matched。为了排除尺度影响还是做了，结论相同。
- **干预确实生效**：去均值 RMS 正好放大了 2 倍、4 倍，跨题 cos 从 0.999 降到 0.984。
- **gain 没有随 α 增大**：5 种条件的 gain 都在 ±1e-4 以内，95% CI 全部包含 0，正负号随机，相对 CE 的变化不到 0.04%。

## KL 噪声底（补充实验，同一题、只用 correct latent，前 16 token）

| KL(α=1 ‖ ·) | 均值 |
|---|---|
| α=1.001（可忽略的扰动，作为 bf16 数值底） | 2.21e-4 |
| α=2 | 2.23e-4 |
| α=4 | 2.13e-4 |
| α=0（去掉残差，每题都只用 μ_t） | 2.03e-4 |
| zero latent（连共同分量也去掉） | 1.90e-3 |

- 把题目残差放大 4 倍、去掉残差、或换成别的题的 latent，引起的 KL 与“α 只改 0.1%”的数值扰动一样大。Executor 对题目相关残差**没有任何可测的响应**。
- 只有去掉共同分量（zero latent）时 KL 才明显升高，约为噪声底的 9 倍。可见 Executor 只对 latent 的公共部分有微弱反应，而且对 CE 几乎没有影响（之前的诊断：zero 和 learned 的 CE 差约 6e-5）。

## 解释与局限

- **结论是 Case C，不是 Case B。** Case B 要求 logits 有可测的变化，而这里 KL 没有超过数值底。
- 题目残差并不是“被公共向量掩盖了”，而是 Executor 根本不读它。即使把残差放大 4 倍、跨题 cos 降到 0.984，Executor 仍然不读。
- 这与上一轮诊断（step swap 在任何位置都无影响）一致。
- **局限**：
  - α 最大只到 4。z_out 跨题差异原本只有 token 间差异的 1/21 到 1/40，α=4 后仍比 token 间差异小 5–10 倍。按任务限定没有测更大的 α。
  - 只测了一个 checkpoint 和 128 道评估题。
  - 只用 teacher-forced CE 和 KL，没有做生成评测。
- 不能把任何结果归因于 Plan 语义。

## 下一步建议（未执行，供讨论）

- 不建议靠调 projector scale 或做残差放大就开正式训练。
- 问题更可能出在执行端的使用方式：三 Loss 训练中，Executor 可以直接从 Question 预测 Execution，latent 被 bypass。
- 下一步应讨论如何让 Executor 的预测必须依赖 Z（训练目标或接口设计），例如先确认 alignment / exec loss 的梯度是否在 Z 上形成了“读取”的压力。
- 本任务到此为止。没有启动 Plan-Bridge、Contrastive Loss、Fine-tune 或 P1。
