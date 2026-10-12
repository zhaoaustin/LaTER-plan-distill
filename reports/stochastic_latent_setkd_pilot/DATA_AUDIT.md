# DATA_AUDIT

- **来源**：`stage2b_executor_50step/data/pairs.jsonl`，即之前 Bridge 构建的同题双 Plan。这些都是 Stage-1 的采样结果，两条参考 Execution 都经 math_verify 判对。
- **规模**：28 道不同的题，20 道训练、8 道 held-out。规范化 Question 后，训练与 held-out 的交集为 0。
- **held-out 8 对**：沿用 Stage-2B 的人工 label audit（`stage2b_executor_50step/evaluations/label_audit.jsonl`）。
  - 8/8 方法不同，参考 Execution 都按 Plan 执行；文字 Plan 对照下 8/8 方法跟随 Plan。
  - 00309 两种方法很接近（Euler φ vs 4 mod 9 的阶），00486 的 Plan A 写得较含糊。
- **训练 20 对**：本轮人工逐对阅读 P_A/P_B（详见 METHOD_SPAN_AUDIT.md）。
  - 19 对的方法不同；00471 的两条 Plan 本质是同一方法（圆心 (r,2) 加上点 (8,0) 在圆上），整对排除。
  - 00439（距离公式 vs 点到直线）、00281 较接近，保留，但不进入 Choice loss。
- **Choice 可用的 pair**（D 片段能直接看出方法）：训练 8 对，held-out 5 对（00336、00574、00569、00305、00486）。
- **Teacher 语义分层**：
  - held-out 一侧引用 2A 的自由生成审计：清楚切换 3/8（00336、00574、00486），单边 3/8（00569 B、00507 A、00305 A），部分 1/8（00479），失败 1/8（00309）。
  - 训练 20 对没有 0.6B 自由生成审计，只有 2A 的配对 Δ：extended20 中 18/20 两边都为正。
  - 结果对 8 个 held-out 全部报告，没有挑选。
- **预训练曝光**：2A Encoder 只在 3000 条 plan-only 数据上训练，构建时已排除这 28 道 Bridge 题。因此 2A 没有见过这 28 题的 Plan，但 Compressor 和 2B Adapter 的选型都参考过 held-out 8 的指标，属于间接曝光。
- **规模限制**：总计只有 19+8 道题，只能用于可学性 Pilot，任何结果都不能作为方法泛化的证据。本地没有其他经过核验的双 Plan 数据，也没有合成数据凑数。
- **Execution 长度**：p50 约 800 token，最大 2212 token，没有截断，也没有过滤。
- **AIME24/25 没有使用**。
