# Step 1：冻结 Executor，提高 ranking 权重 + 两个对照（Executor 全程冻结）

## A. Bridge 加大 ranking 权重（50 步，其余设置与主 pilot 相同；`lam1` / `lam10` / `rankonly`，见 `summary.json`）

| 运行 | held-out margin | train margin | CE（正确配对 / A、B 互换） | cos(Z_A, Z_B) | l_rank |
|---|---|---|---|---|---|
| λ=1 | +0.00003 | +0.0016 | 0.174 / 0.174 | 0.9985 | 1.385 |
| λ=10 | −0.00011 | +0.0003 | 0.207 / 0.207 | 0.9980 | 1.386 |
| 只用 ranking | −0.011 | +0.0075 | 5.03 / 5.02 | 0.9985 | 1.393 |

- 三组都没有信号。ranking loss 停在随机水平 2·ln2，而且 **Bridge 输出的 Z_A 与 Z_B 几乎相同**（cos ≥ 0.998）。
- 只用 ranking 时，Z 被推出正常分布，CE 崩坏，但 Z_A 和 Z_B 仍然分不开。

## B. 自由 Z 对照（`free_z/`）：跳过 Bridge，每题直接优化 Z_A、Z_B

- 设置：8 道训练题，Adam 60 步，每个向量投影到 token RMS，λ=1。
- 结果：margin 从约 0 升到 **1.06**（16/16 为正）；CE 从 0.262 **降到** 0.217；cos(Z_A, Z_B) ≈ 0。
- 文字 Plan 对照的 margin 为 0.167，自由 Z 是它的 6 倍多。
- **结论：冻结的 Stage-1 Executor 可以通过 8 个 embedding slot 被强烈地、有方向地引导。** 接口本身没问题。
- 注意：这里是逐题过拟合，Z 很可能直接编码了具体的 Execution 文本。它只证明 slot 的容量和可控性，不证明 Plan 语义或泛化。

## C. Bridge 输入的可分性（`encoder_state_check.json`，冻结编码器、28 对、mean-pooled Plan 状态）

| 第 27 层 | 同题 A vs B | 不同题 |
|---|---|---|
| 原始向量 cos | 0.975 | 0.925 |
| LayerNorm 后（Bridge 实际看到的） | 0.975 | 0.925 |
| 去掉前 8 个离群维 | 0.890 | 0.668 |
| 跨题去均值 | 0.642 | −0.049 |

- 前 8 个离群维（massive activations）占向量范数的 **89%**。LayerNorm 去不掉它们，所以 Bridge 看到的 A、B 输入几乎相同。
- 其次，题目内容又占了剩余差异的大部分：去均值后同题 A/B 的 cos 仍有 0.64。
- Plan 之间的差异存在，但被淹没了。其他层的结果相同（第 9、18、36 层）。

## 结论（修正）

- 瓶颈**不在冻结的 Executor**，而在 **Bridge 的输入表示与结构**。
  - Executor 可以被引导（B）。
  - Bridge 的输入被离群维和题目内容主导，A/B 差异很小（C）。
  - 加上 CE 项对“通用前缀”的拉力，Bridge 收敛到与 Plan 无关的 Z（A）。
- 因此在考虑 LoRA 之前，应先修 Bridge，例如：
  - 输入按维度标准化（用训练集 Plan 状态的均值和方差，或直接去掉离群维）；
  - 减去只读 Question 的状态，只保留 Plan 带来的增量；
  - 用 token embedding 或更浅的层作为输入；
  - 适当加大学习率。
- 这些都属于冻结 Executor 下的廉价诊断，尚未执行。
