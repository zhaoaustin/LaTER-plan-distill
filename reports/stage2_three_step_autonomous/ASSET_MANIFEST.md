# ASSET MANIFEST — Stage 2 三步版（2A 冻结 → 2B Latent KD → 2C Control vs Choice）

运行目录：`/scratch/hz33227/e2c/stage2/outputs/stage2_three_step_autonomous_20261012/`（新目录，没有覆盖任何旧 checkpoint）

## 环境
- 机器：Slurm 作业 48907334，节点 e4-1，2× NVIDIA B200（183 GB），开始时两卡空闲（4 MiB），没有其他训练进程。
- Python 环境：`~/.conda/envs/later-b200`（torch 2.8.0+cu128，transformers 5.12.1，peft 0.18.1），bf16 autocast。
- 代码：仓库 `/scratch/hz33227/e2c/stage2/LaTER`，基于 commit cec1b0c（分支 exp/stochastic-latent-setkd），新增 `later/scripts/stage2t/three_step.py`。Student 类、method-span 构造（`build_pairs`）、span 审计常量（CHOICE_VALID / EXCLUDED）都从 `later/scripts/stage2s/stochastic_pilot.py` 复用；本轮噪声恒为 0，不用 Set Matching。

## 模型
| 资产 | 路径 / 版本 | 本轮用途 |
|---|---|---|
| Stage-1 Qwen3-4B E²C SFT | `/scratch/hz33227/e2c/Explore-Execute-Chain/outputs/e2c_qwen3_4b_sft/best_hf` = HF `zhaohq/Qwen3-4B-E2C-SFT-only` @ a6b0e2fe（step 800） | Student 主干（冻结） + 一套 LoRA r16（q/v/o）+ LaTER LatentProjector |
| LaTER Recurrent Projector | `later/src/.../load_stage2_model`，`latent_output_rms="token_embedding"`；Stage-1 checkpoint 里没有 projector 权重（MISSING → 新初始化，seed 0） | 自主 z1..z8 递归生成 |
| Stage 2A Compressor step 300 | `/scratch/hz33227/e2c/stage2/outputs/stage2a_external_plan_ae/checkpoints/step300.pt`（sha256 7c63e24d…） | 冻结 Teacher；Z_T 来自 2B 的离线缓存 `stage2b_executor_50step/data/z_cache.pt`（上一轮已验证与在线重算最大差 2.8e-5） |
| Stage 2A Reader（Qwen3-0.6B @ c1899de + LoRA + 2560→1024 adapter，同一 step300.pt） | HF cache | 只读：Teacher 语义审计、对自主 Z_S 的可读性诊断 |
| Stage 2B step-50 Adapter | `stage2b_executor_50step/checkpoints/step50.pt` | **不使用**（它是用训练对的 Execution 训练的，上一轮已发现会造成训练题泄漏） |
| 旧 Stochastic pilot checkpoints | `stochastic_latent_setkd_pilot_20261012/checkpoints/*` | **不使用**（2B 从 Stage-1 + 新 projector 起步） |

## Z_T ↔ Z_S 接口
- 2A Compressor 的输出逐向量 RMS 校准到 4B token-embedding RMS（0.0219532）；2A reader 读入方式是 `z / 0.0219532`。所以 Z_T 本来就定义为“写进 4B 输入 embedding 槽的 8 个软 token”。
- Student 的 Z_S：projector 输出同样逐向量 RMS = token RMS，写进同一位置的 8 个槽：`chat(Q) + '<EXPLORATION>\n' + [z1..z8] + '</EXPLORATION><EXECUTION>\n'`。
- 因此 KD 直接在这个空间做 MSE，没有额外的 adapter 或校准层；同一套接口用于训练和评估。
- 本轮 L_KD = mean((Z_S − sg(Z_T))²) / tok_rms²。也就是 spec 中的 MSE 乘以常数 1/tok_rms²，使 Z_S=0 时 loss=1，正交随机向量时约为 2。

## 数据（全部复用 2B 已构造的文件，sha256 前 16 位）
- `train_singles.jsonl`（e1d69cd6978ebee0）：2000 条，2000 道不同的题，每题一条 Plan（因此不存在同题多 Plan 的 KD 冲突）。
- `val_singles.jsonl`（152f79b315c22899）：100 条，题目全部不同；评估取前 48 条。
- `pairs.jsonl`（d4f503512e6c18d7）：28 对 = 20 对训练 + 8 对 held-out。
