# TEACHER SEMANTIC AUDIT（2A step 300，冻结，只读）

**判定：TEACHER_SEMANTICS_WEAK（方法层面）。** Plan 重建层面合格，Teacher 编码没有坏，所以不是 TEACHER_ENCODING_BROKEN；按任务书只允许做受限的诊断性 pilot。

## 已有证据（复用 2A 报告，未重跑）
- 未见 Plan 上的重建：打乱 Z 比正确 Z 的 CE 高 0.402 [0.379, 0.426]，300/300 为正；均值 Z 和零 Z 只生成模板。
- 同题 A/B 的 CE 配对：held-out 8/8 两边都为正，扩展 20 对 18/20。
- 只看方法类 token 时信号弱很多：held-out 平均 Δ 只有 0.040，8 对中 4 对的 B 方向为负。
- held-out 自由生成（人工）：清楚切换 3/8，部分恢复 4/8，失败 1/8。

## 本轮补充（此前没覆盖的 20 对训练双 Plan，reader 自由生成 320 token）
- 自动判定（生成文本的 4-gram 更接近哪条 Plan）：两边都更接近各自 Plan 的 10/20 对，40 条中 30 条单侧正确；held-out 自动判定 6/8，比人工判定的 3/8 宽松。
- 8 个 Choice 有效训练对：5 个 AUTO_PASS，3 个 AUTO_PARTIAL。
- 局限：训练对只有自动判定，没有人工复核；4-gram 接近度有一部分反映的是措辞，不完全是方法。

## 新增：Teacher Z 对 Stage-1 Executor 的方法控制（未训练，raw Z_T 直接写进 8 个槽）
- 4 格 Method Choice 两个方向都正确：held-out 0/8，训练对 0/19。
- m(Z_TA) − m(Z_TB)：held-out 均值为 −0.009（1/8 为正），训练 19 对为 −0.001。
- 也就是说，没经过训练的 Executor 完全不按外部 Z 选方法，这正是 2C 的 Choice_external 本来要训练的内容。

## 逐对标签
| id | role | Choice span 有效 | reader(Z_A) ov A/B | reader(Z_B) ov A/B | 标签 | 理由 |
|---|---|---|---|---|---|---|
| s1val-00645 | train_pair | 是 | 0.109/0.127 | 0.053/0.188 | AUTO_PARTIAL | 仅自动判定（4-gram 更接近哪条 Plan），未人工复核 |
| s1val-00359 | train_pair | 否 | 0.292/0.077 | 0.011/0.075 | AUTO_PASS | 仅自动判定（4-gram 更接近哪条 Plan），未人工复核 |
| s1val-00499 | train_pair | 否 | 0.032/0.024 | 0.019/0.032 | AUTO_PASS | 仅自动判定（4-gram 更接近哪条 Plan），未人工复核 |
| s1val-00689 | train_pair | 是 | 0.043/0.000 | 0.000/0.076 | AUTO_PASS | 仅自动判定（4-gram 更接近哪条 Plan），未人工复核 |
| s1val-00472 | train_pair | 是 | 0.110/0.077 | 0.047/0.117 | AUTO_PASS | 仅自动判定（4-gram 更接近哪条 Plan），未人工复核 |
| s1val-00422 | train_pair | 否 | 0.089/0.071 | 0.031/0.031 | AUTO_PARTIAL | 仅自动判定（4-gram 更接近哪条 Plan），未人工复核 |
| s1val-00683 | train_pair | 否 | 0.272/0.146 | 0.106/0.035 | AUTO_PARTIAL | 仅自动判定（4-gram 更接近哪条 Plan），未人工复核 |
| s1val-00626 | train_pair | 是 | 0.240/0.200 | 0.066/0.099 | AUTO_PASS | 仅自动判定（4-gram 更接近哪条 Plan），未人工复核 |
| s1val-00496 | train_pair | 是 | 0.083/0.053 | 0.074/0.136 | AUTO_PASS | 仅自动判定（4-gram 更接近哪条 Plan），未人工复核 |
| s1val-00344 | train_pair | 是 | 0.055/0.041 | 0.042/0.042 | AUTO_PARTIAL | 仅自动判定（4-gram 更接近哪条 Plan），未人工复核 |
| s1val-00660 | train_pair | 否 | 0.120/0.054 | 0.018/0.018 | AUTO_PARTIAL | 仅自动判定（4-gram 更接近哪条 Plan），未人工复核 |
| s1val-00368 | train_pair | 否 | 0.061/0.087 | 0.042/0.132 | AUTO_PARTIAL | 仅自动判定（4-gram 更接近哪条 Plan），未人工复核 |
| s1val-00535 | train_pair | 是 | 0.403/0.016 | 0.000/0.060 | AUTO_PASS | 仅自动判定（4-gram 更接近哪条 Plan），未人工复核 |
| s1val-00491 | train_pair | 是 | 0.355/0.183 | 0.146/0.093 | AUTO_PARTIAL | 仅自动判定（4-gram 更接近哪条 Plan），未人工复核 |
| s1val-00653 | train_pair | 否 | 0.125/0.067 | 0.020/0.050 | AUTO_PASS | 仅自动判定（4-gram 更接近哪条 Plan），未人工复核 |
| s1val-00281 | train_pair | 否 | 0.093/0.121 | 0.175/0.320 | AUTO_PARTIAL | 仅自动判定（4-gram 更接近哪条 Plan），未人工复核 |
| s1val-00439 | train_pair | 否 | 0.084/0.076 | 0.107/0.131 | AUTO_PASS | 仅自动判定（4-gram 更接近哪条 Plan），未人工复核 |
| s1val-00498 | train_pair | 否 | 0.014/0.056 | 0.100/0.220 | AUTO_PARTIAL | 仅自动判定（4-gram 更接近哪条 Plan），未人工复核 |
| s1val-00313 | train_pair | 否 | 0.134/0.077 | 0.128/0.106 | AUTO_PARTIAL | 仅自动判定（4-gram 更接近哪条 Plan），未人工复核 |
| s1val-00471 | train_pair | 否 | 0.204/0.194 | 0.131/0.168 | AMBIGUOUS | 两条 Plan 是同一方法（已整对排除） |
| s1val-00336 | heldout8 | 是 | 0.238/0.171 | 0.099/0.166 | PASS | 2A 报告人工：清楚切换 Lagrange vs 不等式 |
| s1val-00574 | heldout8 | 是 | 0.048/0.018 | 0.009/0.075 | PASS | 人工：清楚切换 因式分解 vs 试值 |
| s1val-00569 | heldout8 | 是 | 0.092/0.000 | 0.012/0.217 | PARTIAL | 人工：只恢复一侧/部分 |
| s1val-00507 | heldout8 | 否 | 0.126/0.019 | 0.146/0.035 | PARTIAL | 人工：只恢复一侧/部分 |
| s1val-00479 | heldout8 | 否 | 0.090/0.056 | 0.022/0.089 | PARTIAL | 人工：只恢复一侧/部分 |
| s1val-00309 | heldout8 | 否 | 0.032/0.008 | 0.085/0.103 | FAIL | 人工：两边都没恢复 |
| s1val-00305 | heldout8 | 是 | 0.202/0.090 | 0.194/0.194 | PARTIAL | 人工：两边都走代入 |
| s1val-00486 | heldout8 | 是 | 0.097/0.065 | 0.000/0.052 | PASS | 人工：清楚切换 等差余弦公式 vs 单位根 |
