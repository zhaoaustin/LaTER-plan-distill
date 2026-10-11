# Step 1：冻结 Executor，提高 ranking 权重（50 步，其余设置与主 pilot 相同）

## 运行条件
- `lam1`：λ_rank = 1
- `lam10`：λ_rank = 10
- `rankonly`：去掉 CE，只用 ranking 项
- 共同设置：同一 seed、初始化和 batch 顺序，lr 1e-4。
- 本轮只用 margin 评估，没有做自由生成：held-out 上没有信号，生成没有意义。
- 逐步的 eval 记录在各子目录的 `train_log.jsonl`；汇总在 `summary.json`。

## 结论：冻结 Executor 在 embedding 层不可被引导（50 步内）
- **λ=1 和 λ=10**：ranking loss 停在随机水平 2·ln2（约 1.386），held-out margin 约为 0。λ=1 时 train margin 只升到 0.0016，不到文字 Plan 对照（0.167）的 1%。
- **只用 ranking**：Bridge 把 Z 推出了正常分布，Execution CE 从 0.22 升到 5.0。margin 变成正负随机的大噪声，ranking loss 仍然没有下降。也就是说，Z 能破坏 Executor 的输出，却无法把它引向指定的方法。
- **建议**：下一步只能让 Executor 本身参与适应（LoRA，需要用户批准）。
- **局限**：只试了 50 步、一个 lr、一种 margin 定义（对整段 Execution 取平均）。
