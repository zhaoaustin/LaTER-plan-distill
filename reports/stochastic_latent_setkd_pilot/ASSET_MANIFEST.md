# ASSET_MANIFEST（全部只读，新输出只写入本目录）

| 项目 | 实际路径 / 值 |
|---|---|
| Repo | `/scratch/hz33227/e2c/stage2/LaTER`，分支 `exp/stochastic-latent-setkd`（基于 8869613 = Stage 2B 报告提交）；本轮代码 `later/scripts/stage2s/stochastic_pilot.py`（sha256 见末尾） |
| 环境 | `~/.conda/envs/later-b200`：torch 2.8.0+cu128、transformers 5.12.1、peft 0.18.1、CUDA 12.8 |
| GPU | Slurm 作业 48907334，节点 e4-1，2×NVIDIA B200（启动前 `nvidia-smi`：两卡各 4 MiB、0% 占用，无其他本项目进程） |
| Student 初始化 | Stage-1 step 800：`/scratch/hz33227/e2c/Explore-Execute-Chain/outputs/e2c_qwen3_4b_sft/best_hf`（= HF zhaohq/Qwen3-4B-E2C-SFT-only @ a6b0e2fe）。没有选 2B-v2 初始化，因为 2B-v2 没跑过；也没有选 2B，因为它的结论是 NO_PLAN_CONDITIONING_SIGNAL |
| LaTER Projector | `later/src/train/modeling_latent.py::LatentProjector`（Linear-GELU-Linear，输出按 4B token-embedding RMS 0.021953 归一化，对应 `load_stage2_model(latent_output_rms="token_embedding")`）。**新初始化**（seed 0）：旧 projector 都出自全参 Stage-2 run（`stage2/runs/*`），当时 backbone 也被改动，与 Stage-1 权重不兼容 |
| 共享 LoRA | 一套，r16/α32，q/v/o，dropout 0（973 万参数）；Planner 递归与 Executor 共用 |
| Teacher Encoder | Stage-2A step300 Compressor：`stage2/outputs/stage2a_external_plan_ae/checkpoints/step300.pt`（sha256 7c63e24d…bbb6），Z 来自 Stage-2B 的已验证缓存 `stage2b_executor_50step/data/z_cache.pt`（sha256 24fa84c2…a866，离线重算的最大误差 2.8e-5） |
| Teacher 映射 T | 冻结的 Stage-2B step-50 Adapter（LN + Linear + 尺度；实测 W 与单位阵的最大偏差 0.0032，尺度 0.02197），`stage2b_executor_50step/checkpoints/step50.pt`（sha256 4b9221cc…7519）。2B 的 Executor LoRA **没有**加载 |
| 0.6B Decoder | 本轮没有加载 |
| 双 Plan 数据 | `stage2b_executor_50step/data/pairs.jsonl`（sha256 e1d69cd6…f2e4）：20 个训练对和 8 个 held-out 对，每题都是不同的问题 |
| Teacher 统计量 | 2000 条训练单样本（`train_singles.jsonl`，sha256 152f79b3…a75a9f）的 T(Z) 逐维 RMS，只用训练集 |
| 本轮 checkpoint | `checkpoints/step0.pt`（a2a71b77…d782）、`after_tiny_overfit.pt`（sha256 6f3029a58e2c70d0b58aed48f677c9c168db2b3bb8259239221921cd094d82be）、`step25.pt`（与 after_tiny 同一步） |
| 代码 sha256 | stochastic_pilot.py 1a28c700a5e12e68fc2b78215f76123f69127231ccb6420192ed6c830decf0a6 |
| 参考任务书 | 在服务器上没有找到 `Stage2B_v2_*`、`Stage2_Overnight_*` 等 md 文件；2B-v2 和 Overnight 任务都没有运行记录 |
