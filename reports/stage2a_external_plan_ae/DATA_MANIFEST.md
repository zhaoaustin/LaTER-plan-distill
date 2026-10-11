# Stage 2A 数据清单

## 模型与环境
- **Stage-1 编码器**：`/scratch/hz33227/e2c/Explore-Execute-Chain/outputs/e2c_qwen3_4b_sft/best_hf`
  - step 800，对应 HF zhaohq/Qwen3-4B-E2C-SFT-only @ a6b0e2fe。
  - 36 层，hidden 2560；本次取第 27 层。
- **Decoder**：Qwen/Qwen3-0.6B @ c1899de2，从 HF 下载并固定版本；hidden 1024。
- **分词**：0.6B 与 4B 对 Plan 文本的分词结果完全一致（抽查 200 条）。
- **硬件与库**：1× B200（Slurm 48907334）；torch 2.8.0+cu128，transformers 5.12.1，peft 0.18.1。

## 数据来源
- **训练集**：`/scratch/hz33227/e2c/stage2/data/stage2_train.parquet`（46510 条）
- **验证集**：`/scratch/hz33227/e2c/stage2/data/stage2_val.parquet`（2446 条）
- **Plan 字段**：取 `<EXPLORATION>` 内的 plan 字段，不含 Execution。
- **规范化**：只删掉 100% 记录都有的开头一句 “I need to make a step-by-step plan.”。
- **字段**：train/val 文件里只保存 record_id、question_hash 和 Plan，**不含 Question 字段**。

## 采样与过滤（seed 42）

| | train | val |
|---|---|---|
| 原始记录 | 46510 | 2446 |
| 重复题目去除 | 90 | 2 |
| 与评测集精确重复去除 | 0 | 0 |
| Bridge 的 28 道题去除 | 0 | 28 |
| 超过 288 token 过滤 | 1821（3.9%） | 90（3.7%） |
| Plan token 数 p50 / p90 / p95 / max（过滤前，含 EOS） | [159.0, 248.0, 278.0, 501] | [161.0, 248.5, 275.25, 501] |
| **采样** | **3000** | **300** |

- **长度上限**：如果用任务书默认的 192，会过滤约 28% 的样本（p90 = 245），所以按规则改为 288（约等于 p95）。超长样本直接过滤，**不截断**。
- **题目隔离**：train 与 val 的题目交集为 0；sha256 见 data_manifest.json。
- **评测集重复检查**：对 AIME24/25、AMC23、MATH-500、Minerva、OlympiadBench、GSM8K 做了规范化后（小写、合并空白）的精确匹配，命中 0 条。**近似重复没有检测**，这是局限。
- **同题双 Plan 评估集**：Bridge 实验的 8 对 held-out，加上另外 20 对（原 Bridge 训练对）作扩展集。
  - 28 道题都来自 stage2_val，已从本次 train 和 val 中排除。
  - 8 道 held-out 题在 Stage-2 训练集中出现 0 次。
  - 只用 P_A 和 P_B，不用 Question 或 Execution。
- **改写集**：6 条人工写的改写，只改措辞、不改方法（`data/paraphrases.jsonl`）。

## 编码器缓存
- 3362 段文本：train 3000、val 300、配对 56、改写 6；共 471666 个训练 token。用时 47 秒。
- **输入**：只有 Plan 文本，不加对话模板，不拼 Question，不拼 Execution。
- **4B 权重**：缓存前后参数和不变（True）。训练进程从不加载 4B。
- **标准化**：σ 下限取第 5 百分位（1.442），被抬到下限的维度占 5%；截断阈值 ±5。统计量只用训练集计算。
