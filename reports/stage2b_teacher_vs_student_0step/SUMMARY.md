# SUMMARY — Stage 2B step 50：Executor 能不能读 Teacher latent？（0 步，8 道 held-out）

## 结论：**Teacher latent 读不出来**；同一个 Executor 换成文字 Plan 能 8/8 切换方法。

| 诊断项 | 结果 |
|---|---|
| 接口检查（加载、shape、尺度、slot、KV/位置） | **PASS** |
| Teacher 2×2 的 delta_match 均值 [95% CI] | **−0.0002 [−0.0048, +0.0040]**；delta_A>0 有 2/8，delta_B>0 有 4/8，delta_match>0 有 3/8（完整 E 上：−0.0002，3/8） |
| 注入 Teacher Z_A / Z_B 后 greedy 生成的方法切换 | **0/8**（人工核对）。自动标签：TA = BBBABBAB，TB = BBBABBAB；2 题文本逐字相同。正确率 6/8 和 7/8 |
| Student 与零 latent 的 ΔNLL；跨题 cos、方差 | NLL(zero) − NLL(student) = −0.0001 [−0.0005, +0.0002]；48 道未见题上跨题 cos 0.996，方差比 0.007，**仍然坍缩** |
| 同一 Executor 的文字 Plan oracle | **方法切换 8/8**（自动标签 PA = AAAAAAAA，PB = BBBBBBBB；人工核对 5 题切换明显，3 题由方法词佐证）。同口径的 delta_match = **+0.135 [0.099, 0.178]**，8/8 为正。正确率 7/8 和 6/8 |
| Teacher A/B 的差异；Teacher 与 Student 的尺度 | 同题 cos(Z_A, Z_B) 平均 0.69，相对 L2 距离 0.78，方向差异明显；三者每个 slot 范数都是 1.1108，尺度相同 |

- **人工核对**：自动的 n-gram 标签器有误判。例如 00336 题注入 TA/TB 后其实都是 Lagrange 方法，却被标成了 B。所以以人工判断为准：注入 Teacher A/B 后，每题两条生成用的都是同一种方法；文字 Plan 下，方法分别跟着 P_A / P_B 走。
- **对比**：文字 oracle 的 delta_match 大约是 Teacher latent 的 600 倍（0.135 对 0.0002）。两者输入长度不同，这个倍数只能看量级，不能当作严格可比。

## 按任务书的判断逻辑
**Teacher A/B 不能控制方法，但文字 oracle 能控制。**
- 主要问题在 **latent → Execution 的读取（Bridge / Executor 读出）**，不只是 Student Projector 坍缩。
- 即使 Student 生成出了和 Teacher 一样的 latent，当前的 Executor 也不会照着它执行。
- 接口和尺度已经检查过，没有发现不匹配，所以“分布不匹配”可以基本排除。单靠调大 Anchor JSD，不能让它学会读 Plan。

## 下一步唯一建议（需要你确认，不会自动启动训练）
**先单独训练“Executor 读 Teacher latent”这一能力，暂时不管 Student。**
- 冻结 2A Encoder 和 Projector，只训练 LoRA。输入为 Z_T(P)，目标为对应的 E。
- 以 19 对训练题的同题 A/B 对比为主项：matched CE 加上 matched−swapped 排序。
- 用 8 道 held-out 的 2×2 delta_match 和 A/B 生成切换来验收。
- 只有这一步能在 held-out 上切换方法，再回头修 Student 的坍缩才有意义。

## 局限
- 只有 8 对，属于快速诊断，不做显著性结论。
- 方法标签：自动判定 + 人工粗读开头，不是逐步审阅全文。
- Teacher A/B 方向不同，也不等于它们编码了正确的 Plan 语义。
