# GATE_REPORT

| Gate | 结果 | 关键数字 |
|---|---|---|
| **G0 资产 / 进程** | PASS | 两卡 B200 空闲；Stage-1 step800、2A step300、2B step50 adapter、28 对数据都已定位并记录 checksum（ASSET_MANIFEST.md）。2B-v2 和 Overnight 任务没有运行记录 |
| **G1 Teacher / 数据 / 接口** | PASS（有说明） | T_A ≠ T_B（28 对没有同 key）。同题 A/B 归一化距离均值 0.564，跨题 1.042；cos_AB 为 0.58–0.86。Teacher RMS 0.0234，Student 每个 slot 的 RMS 0.02195（= token RMS），两者写入同一组 8 个 input-embedding slot。说明：T 用的 2B adapter 在 2B 中没有被证明能驱动 Executor，这里只当作冻结的空间映射 |
| G2.1 σ=0 确定性 | PASS | 两次生成的最大差为 0.0 |
| G2.1 噪声传播 | PASS | σ=0.1 时 z1 噪声 RMS 比为 0.101；两个 ε 在 z1–z8 上的相对差为 [0.14, 0.14, 0.07, 0.09, 0.10, 0.78, 0.92, 0.53]；‖∂z8/∂ε‖=4.92 |
| G2.2 同题共享 Q | PASS | K 条路径使用同一组 prompt ids，只有 ε 不同 |
| G2.3 SetKD 置换不变 / 重新匹配 | PASS | 交换 student 顺序、交换 teacher 顺序，结果都是 2.03203（完全一致）；student 样本顺序反过来时会正确重新匹配 |
| G2.4/5 三项 loss 的梯度 | PASS | Projector / LoRA 的梯度范数：set 153.3 / 19.7，choice 9.2 / 1.2，exec 63.8 / 8.4，全部有限且非零。**诊断：乘上 λ 之后，choice 的有效梯度（0.92）约为 set 的 1/170，基本被 SetKD 掩盖。** 按任务书规定不改系数 |
| G2.6/7 Mask / span / 前缀 | PASS | C+D 是 target 的前缀；短序列与长序列在 D 片段上的 logprob 完全一致（−0.56317）；prompt 中不含 P/E 文本；C 是中性的 “1. Use the” |
| G2.8 干预时重建 KV | PASS | 所有评分 forward 都设 use_cache=False；zero Z 会改变 logits，且结果有限 |
| 训练路径 = 生成路径 | PASS | 首 token 的 argmax 一致 |
| G2.10 Save / Reload | PASS | reload 前后 loss 相同（2.321343 / 2.321343）；冻结参数保持不变 |
| Smoke（3 步，计入预算） | PASS | 1.56 s/step（每卡 2 题），峰值 50 GB |
| **G3 Tiny Overfit（12 对，25 步）** | **FAIL → TRAIN_FIT_FAIL / FAIL_COLLAPSE** | 同题两个样本的距离 0.21→0.003（z8 cos 0.906→0.9999），teacher A/B 距离约 0.6；SetKD 2.01→1.02，降到“两个样本都放在 A/B 中点”的解；Choice 一直约为 0.69（ln2）；Δ_A/Δ_B 在 ±0.007 内随机波动，两边都为正的比例没有上升。详见 FINAL 报告 |
