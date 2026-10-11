# Stage 2B 数据与模型清单

## 模型
- **Executor**：Stage-1 Qwen3-4B E²C SFT step 800，路径 `/scratch/hz33227/e2c/Explore-Execute-Chain/outputs/e2c_qwen3_4b_sft/best_hf`（= HF zhaohq/Qwen3-4B-E2C-SFT-only @ a6b0e2fe）。hidden 2560，36 层。
- **Stage-2A 编码器**：
  - checkpoint：`/scratch/hz33227/e2c/stage2/outputs/stage2a_external_plan_ae/checkpoints/step300.pt`（sha256 7c63e24d5dd4a95357208a4d0dc2726eb7d66d94b2b43035639b01087f38bbb6）。
  - 只使用其中的 Compressor（8 个 query，2 层）和训练集标准化统计量。
  - Plan 特征来自冻结 4B 第 27 层的 Plan-only 缓存（`stage2a_external_plan_ae/data/plan_states_L27.pt`）。
  - **身份验证**：重新计算 held-out 配对 Δ，得到 0.109789，与报告值相同。
  - **Qwen3-0.6B Decoder 和 2560→1024 的 Adapter 只在验证这一步加载，不参与 2B 的训练图，也不用于生成 Z。**
- **没有加载**：旧的 Recurrent Projector 和旧 Bridge 都没有加载。

## 数据
- **来源**：`/scratch/hz33227/e2c/stage2/data/stage2_train.parquet`（46510 条）和 `stage2_val.parquet`（2446 条）。字段包括 question、plan、execution、answer。
- **训练单样本**：
  - 候选是 Stage-2A 的 3000 条训练 Plan，按 record_id 与 stage2_train 对应，取出 Q、E、A。
  - 过滤掉 Plan 中含有最终答案字符串的 135 条，剩 2865 条，再用 seed 42 采样 **2000** 条。
  - Execution 长度 p50 / p95 / max 为 [795.0, 1763.199999999999, 1993] token，没有截断，也没有因长度过滤。
- **验证单样本**：
  - 从 Stage-2A 的 val Plan 中过滤掉含答案的 13 条，再采样 **100** 条。
  - 用于比较正确 Z 与跨题 Z、均值 Z、零 Z。
- **同题双 Plan**：
  - 训练用 20 对（原 Bridge 训练对，用于 ranking）；评估用 8 对 held-out（原 Bridge held-out）。
  - 两条参考 Execution 都经 math_verify 判对。人工审计结果见 `evaluations/label_audit.jsonl`：8 对都有效，文字 Plan 对照下 8/8 方法会跟着 Plan 走。
  - 00486 的 Plan A 写得较含糊，00309 的两种方法比较接近，已标注，但仍计入分母。
- **隔离**：规范化 Question 后，held-out 8 题与训练单样本、验证单样本、20 对训练 pair 之间的交集都为 0；训练与验证单样本之间的交集也为 0。
- **Z 缓存**：2156 段 Plan 的 Z，形状 [8, 2560]，平均 RMS 0.02195。在线重算与缓存的最大差为 2.8e-05（bf16 批次组成造成）。
- **Mean Z**：只用 2000 条训练单样本的 Z 求平均，再按 RMS 归一化。
