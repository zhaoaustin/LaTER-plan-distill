# DATA AUDIT

## 来源
- 全部复用 Stage 2B 已构造的数据（`/scratch/hz33227/e2c/stage2/outputs/stage2b_executor_50step/data/`）。原始来源是 `stage2/data/stage2_train.parquet`（46,510 条）和 `stage2_val.parquet`（2,446 条），与旧记录一致。
- 单 Plan 样本只在 Stage-2A 的 3000 条训练 Plan 中选取，构造时已过滤 Plan 中出现最终答案字符串的样本，并检查了 E 与 P 的步骤对应关系（E 按 P 的编号逐步展开）。

## 规模与题目级隔离（本轮重新验证）
| 集合 | 条数 | 不同题目数 | 用途 |
|---|---:|---:|---|
| train_singles | 2000 | 2000 | 2B KD；2C KD+Exec |
| val_singles | 100（评估用前 48） | 100 | 未见题：KD correct/shuffled、Exec CE 干预、2A reader |
| train_pair | 20（00471 排除，剩 19；Choice 有效 8） | 20 | 2C 中只使用这 8 个 Choice 有效对，每步 2 对 |
| heldout8 | 8（Choice 有效 5） | 8 | 只用于评估 |

- 题目按空白规范化、小写后比较：train∩val=0，train∩heldout=0，train∩train_pair=0，val∩heldout=0，train_pair∩heldout=0；按前 120 字符比较的近重复也都是 0。
- train_pair 中另外 11 对没进入任何训练（Choice 无效对不进 2C，以免引入未经验证的方法标签）。评估时仍按 train_pair 角色报告。
- **没有评估 AIME24/25、MATH-500 等下游 benchmark**，所以本轮也没有做 benchmark 污染检查，不对下游准确率下任何结论。

## 同题多 Plan 冲突
- train_singles 每题只有一条 Plan。
- 每个双 Plan 题固定一个默认 Plan 作为自主 KD 和 Exec 的目标：`sha1(id)` 为偶数取 A，否则取 B。这个选择与结果无关。另一条 Plan 只出现在 Choice_external 的反事实分支中。
- 两个 2C 分支对双 Plan 题的处理完全相同（KD 加 Exec 都用默认 Plan），唯一差别是 Choice 项开不开。

## Method-Decision Span
- 直接复用上一轮人工审计（`stochastic_latent_setkd_pilot_20261012/METHOD_SPAN_AUDIT.md`）：C = E_A/E_B 的 token 级最长公共前缀，D_X = C 之后 E_X 的前 24 个 token。
- 训练对有效 8/19，held-out 有效 5/8；无效对标为 NOT_ELIGIBLE_METHOD_CHOICE。
- A/B 打分用同一个 Q、同一个 C、同一组 slot 位置；长度归一化取 mean log-prob，D_A 和 D_B 都是 24 个 token，长度相同。
- 2B-v2 Method Choice 在服务器上没有任何代码、日志或 checkpoint（上一轮已全盘搜索过），因此视为“未运行”。

## 泄漏检查（Gate 1）
- Student 的 prompt（解码后文本）里不出现 Plan、Execution 片段或方法标签。
- Z_T 只由 Plan 计算，并且 detach。
- 篡改 Teacher 缓存后 Z_S 完全不变。
- Exec 的 loss 只覆盖 `E + </EXECUTION><|im_end|>`。
