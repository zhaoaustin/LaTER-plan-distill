# Stage 2 Latent-Steps Collapse 诊断 — 结论（Forward-only）

**模型**：三 Loss Joint（Exec CE + 0.1 Align + 0.3 Explicit CE），L=8，step 500
`/scratch/hz33227/e2c/stage2/runs/stage2_L8_rms_2900/checkpoint-500`（训练 commit a4f1753，`latent_projector_output_rms = 0.021953`（token_embedding），projector dropout 0）。
**数据**：`stage2_val.parquet` 前 128 条（= 训练时 probe 集，128 道不同题目），先跑了 16 条 dry run。
**模式**：`eval()` + `inference_mode()`，fp32 权重 + bf16 autocast（与训练 probe 一致）；无 optimizer、无 backward、未改任何 Checkpoint/Loss/训练代码。只新增两个诊断脚本，hook 是只读的。

## P0 Gate（全部通过）

| 检查 | 结果 |
|---|---|
| 加 hook 前后 Execution CE / target logits 最大差 | 0.0 / 0.0 |
| 重复运行 logits 最大差 | 0.0 |
| hook 捕获的 z_out 与 builder 实际写入 slot 的 embedding | 最大差 0.0 |
| h_in[t+1] 与 z_t 之后的 hidden | 最大差 0.0（递归对齐无误） |
| z_out RMS | 0.021941–0.021965（= output_rms，分析的是真实喂回模型的向量） |
| learned Z 经 latent_override 重新喂入 vs 原 builder 路径 | logits 最大差 0.0，CE 差 0.0 |
| batch-1 与 batch-8 提取（padding / 位置）的差异 | 数值噪声 RMS = 跨题标准差的 6–15%（h_in）、10–20%（z_out），所以下面的跨题差异是真信号，不是噪声 |

递归路径确认：z₁ = proj(h_seed)，其中 h_seed 是 `<EXPLORATION>\n` 最后一个 token 处的 hidden；z_k = proj(h after z_{k−1})。h_in 是 `Qwen3Model.last_hidden_state`，**在最终 RMSNorm 之后**。builder 逐 slot 共享 KV cache，与训练和 probe 走同一条路径。

## Test A 主要数值（t = 1 / 2 / 4 / 8）

| 指标 | h_in（Projector 输入） | z_out（实际 Latent embedding） |
|---|---|---|
| 跨题 raw cosine 均值 | 0.980 / 0.982 / 0.981 / 0.983 | 0.998 / 0.9994 / 0.9993 / 0.9994 |
| 相对跨题 std（去均值 / RMS） | 0.142 / 0.134 / 0.139 / 0.130 | **0.046 / 0.025 / 0.026 / 0.024** |
| 共同分量范数 / 平均范数 | 0.990 / 0.991 / 0.990 / 0.992 | 0.999 / 0.9997 / 0.9997 / 0.9997 |
| centred effective rank | 21 / 70 / 57 / 65 | 15 / 63 / 51 / 60 |
| CKA(z_out_t, h_in_t) | — | 0.94 / 0.97 / 0.98 / 0.97 |

- **参照（同一批题、同一模型读 gold plan 的显式路径）**：
  - plan 内容 token 处的 hidden，相对跨题 std ≈ 0.83–0.87；固定 token（"1"）处的 hidden ≈ 0.17；
  - gold plan token embedding ≈ 0.95。
  - 换成绝对量：z_out 每维的跨题差异是随机词表 token embedding 之间差异的 **1/21（t=1）到 1/40（t≥2）**。
- **h_out8**（z₈ 之后的 hidden）相对跨题 std 为 0.140，和 h_in₁ 相当，**没有随递归衰减**。
- **周期 2 吸引子**（补充检查）：
  - t ≥ 2 时，同一题内 cos(z_t, z_{t+2}) = 0.996–1.000，而 cos(z_t, z_{t+1}) ≈ 0.28；
  - 奇数步和偶数步的跨题均值向量分成两组，奇偶之间 cos ≈ 0.28–0.33。
  - 也就是说，递归在 t=2 之后进入 A/B 交替的 2-cycle，8 个 slot 实际上只有约 3 个不同的向量（z₁、奇数步、偶数步）。
  - 但题目差异仍随周期保留：去均值后 cos(t, t+2) = 0.6–0.93；用 h_in 按 top-1 找回 step 1 时的同一道题，准确率为 0.54–0.72。
- **共同分量的来源**：z_raw 的共同分量范数约 400–430，题目相关部分约 10–20，输出层 bias 范数只有 0.08。所以共同分量来自 MLP 对 h_in 共享部分的映射，不是 bias。

## Test B：Step swap（受控强制前向，可选）

- **协议**：用 `latent_override` 固定全部 8 个 slot，**swap 之后不重算后续递归**；donor 是固定 derangement（seed 0）。指标是非重复 Execution token 的逐题 CE 差。
- **结果**：swap_step_{1,2,4,8}、prefix 1:{1,2,4}、suffix {2,4,8}:8、all 1:8 共 11 种条件：
  - ΔCE 全部在 −4e-5 到 +8e-5 之间，95% CI 全部包含 0；
  - first-token KL 的均值 ≤ 1e-7；
  - 原始 CE 中位数为 0.18，逐题 |ΔCE| 的中位数为 0.0003。
- 只有 prefix 1:4 的 first-16-token CE 的 CI 不含 0（+0.0009 [0.00004, 0.0018]）。但这是 22 个比较中的一个，量级可以忽略，不作为证据。
- 训练时 step-500 的 probe 结论一致：CE learned 0.236792、shuffled 0.236794、zero 0.236855、no_latent 0.236868。

## 必答问题

1. **z₁ 就塌缩了，还是 z₁→z₈ 中明显趋同？**
   - z₁ 已经高度同质：cos 0.998，相对 std 0.046，绝对差异只有 token 之间差异的 1/21。
   - t=1→2 再降一档（0.046→0.025），t=2..8 基本持平（0.024–0.043，奇偶振荡），**没有单调衰减**。
   - 主要问题出在一开始，加上 t≥2 的周期 2 冗余；不是“递归到 z₈ 才塌”。
2. **塌缩在 Projector 前还是后？两处都有，主体在 Projector 之后。**
   - h_in 的多样性只相当于“固定 token + 不同题目上下文”的水平（0.13–0.17，与显式路径固定 token 处的 0.17 相当），而且不随 t 衰减。
   - Projector 在基本保持几何结构的同时（CKA 0.94–0.99），用一个巨大的共同分量把相对差异再压缩了 3–6 倍。
   - 再经 output_rms 归一化后，题目相关部分只占向量的约 2.5–4.6%。
3. **“8 Latent Tokens 过多”：`NOT_SUPPORTED`。**
   - 没有“早期有信息、后期衰减”的模式：h_in 0.142→0.130，z_out 在 t=2..8 持平，h_out8 为 0.140。
   - 任何位置的 step swap（包括 z₁ 和前缀）对 Execution 都没有可测的影响。执行端对全部 8 个 slot 都不敏感，所以缩短到 4 无从改善“信息被使用”这一点。
   - 周期 2 吸引子说明 8 步确实冗余，但冗余 ≠ 信息衰减。
   - **局限**：只看了一个 checkpoint（三 Loss step 500）、128 题；cosine / std / CKA 只说明几何，不说明语义；Test B 只做了“固定 slot、不重算递归”这一种干预。
4. **现在需要 Fine-tune 吗？`NO`。**
   - 不满足“早期有信息、后期衰减”的前提，**不建议**做 L4 vs L8 训练，也没有启动任何训练。
   - 后续若要改，优先方向是：
     - 执行端 conditioning / 监督，让 Executor 必须依赖 Z：Execution 对 learned/shuffled/zero/no_latent 全部不敏感；
     - Projector 输出中压倒性的共同分量与尺度：z_out 题目相关部分占约 2.5–4.6%，比 token 间差异小 21–40 倍；
     - 周期 2 递归动态：作为次要现象记录。
   - 这些都只是建议，未执行。
5. **用时 / 硬件 / 改动**：
   - GPU 实测：dry run 71 s + 3 次全量各约 84 s，合计约 5.4 分钟；整个任务（读代码、写脚本、调试、报告）约 40 分钟。
   - 硬件：1× NVIDIA B200（Slurm hold job 48907334，节点 e4-1；训练任务处于暂停状态，没有抢占）。
   - 改动：只新增 `later/scripts/train/diag_latent_step_collapse.py` 和 `latent_step_collapse_report.py`。没有修改模型、Loss、配置、训练代码或任何已有 Checkpoint。
   - 全量跑了 3 次：第 2 次补了噪声 gate 指标，第 3 次补存逐题 swap 结果；3 次的数值完全一致。

## 文件

- `metadata.json`：git、checkpoint、数据 hash 与 128 个 record_id、环境。
- `per_step_metrics.csv/json`：t=1..8 的 h_in / z_raw / z_out 及参照的全部指标，含 gate。
- `latent_similarity_by_step.png`：随 step 变化的曲线。
- `examples.md`：10 道不同类型题的逐步对比、样本对曲线和 3 个 swap 案例。
- `step_swap_metrics.json`：含逐题结果。
- `supplementary_cpu_checks.txt`：周期 2 和共同分量来源的补充检查。
- `latents_fp16.pt`、`logs.txt`。
- `dryrun16/`：16 条 dry run 的结果。
- 注意：`per_step_metrics` 中 z_out 对 h_in 的 `nn_retrieval_vs_h_in_same_step` 是跨两个不同向量空间的余弦，**无意义，请忽略**；跨空间只看 CKA。
