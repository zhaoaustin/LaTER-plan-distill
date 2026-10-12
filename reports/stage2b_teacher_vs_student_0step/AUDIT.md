# AUDIT — Teacher vs Student Latent 可读性诊断（0 步，只做推理）

**INTERFACE PASS**（`audit.json`）。全程 0 次 optimizer step、0 次 backward，没有修改任何 checkpoint。

- **Executor**：
  - 基座：`/scratch/hz33227/e2c/Explore-Execute-Chain/outputs/e2c_qwen3_4b_sft/best_hf`（Stage 1 SFT step 800）。
  - 加载：`/scratch/hz33227/e2c/stage2/outputs/stage2b_joint_two_loss_20261012/checkpoints/joint_step50.pt`（step 50，phase joint），216 个 LoRA 张量 + 4 个 projector 张量。加载后与 checkpoint 逐位相同。
  - 状态：eval()，可训练参数 0。**没有加载任何 2C checkpoint。**
- **Teacher Encoder**：`/scratch/hz33227/e2c/stage2/outputs/stage2a_external_plan_ae/checkpoints/step300.pt`。在线重新编码这 16 条 Plan，与 2B 使用的 z_cache 最大差 2.4e-4（16/16 都有缓存的 L27 状态）。
- **尺度与空间**：Z_T^A、Z_T^B、Z_S 每个 slot 的范数都是 1.1108（即 token-embedding RMS 0.02195 × √2560），slot 内方差都约 4.82e-4，均值约 1e-4 到 1e-3 量级。三者处在同一尺度。用的是原 2B 的连续向量注入接口，没有额外投影或归一化。
- **注入**：8/8 题、4 种条件（TA、TB、ZS、zero）都检查过：
  - slot 里确实是对应的向量；
  - 非 slot 位置的 embedding 逐位相同；
  - ids、position、attention mask 相同；
  - 前缀（题目 + 8 个 slot + `</EXPLORATION><EXECUTION>\n`）里没有任何 Plan 文本。
- **自查更正**：第一次运行时，“前缀不含 Plan 文本”这项检查误把 gold Execution 目标也算了进去。Execution 本来就会逐条复述 Plan，所以误报为 FAIL。改为只检查前缀后重跑，所有评分和生成结果与第一次逐位一致。第一次的结果保存在 `run1_prefixcheck_bug/`。
- **数据**：8 道 held-out 同题 A/B（mini bridge 人工审阅数据）。
  - 两条 Execution 的答案都经 math_verify 判为正确，策略不同（人工标签见 `per_question.csv`）；
  - 都不在 train_singles 里。
- **评分区间**：E 的 token-LCP 之后、最后一个 `\boxed` 之前，即策略 span，不含公共的最终答案。8/8 题都找到了 `\boxed`。另外也报告了完整 E 上的结果。
- **生成**：greedy，最多 2048 token，chat template 和 EOS 与 2B 相同。
- **Text oracle**：用同一个 Executor（LoRA 开着），输入为 `chat(Q) + <EXPLORATION>\n + P_X + \n</EXPLORATION><EXECUTION>\n`（Stage 1 的显式格式，P 是原始 Plan 文本），不注入 latent。
