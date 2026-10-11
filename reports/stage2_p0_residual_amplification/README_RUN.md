# P0 Residual Amplification — 复现说明

- **模型**：原始三 Loss Stage 2 checkpoint-500
  - 路径：`/scratch/hz33227/e2c/stage2/runs/stage2_L8_rms_2900/checkpoint-500`
  - Exec CE + 0.1 Align + 0.3 Explicit CE；latent_steps=8；output_rms=0.0219532（token_embedding）
  - 训练 commit：a4f1753b43e968bb8dd4c2a64c6cef6ea1700cfb
  - 初始化：Stage-1 `best_hf`（HF zhaohq/Qwen3-4B-E2C-SFT-only @ a6b0e2fe）
- **诊断代码**：repo `/scratch/hz33227/e2c/stage2/LaTER`，branch `exp/stage2-planrec-two-loss`。
  - 运行时 HEAD 为 f6176ac，在此基础上新增两个脚本（见 `patch.diff`），已提交。
  - 只新增脚本，没有修改模型、Loss、配置或任何 checkpoint。
- **数据**：`/scratch/hz33227/e2c/stage2/data/stage2_val.parquet`，sha256 见 `preflight.json`。
  - 评估集：前 128 条，与之前的 probe 集相同。
  - 校准集：第 128–255 条，与评估集不重叠，题目交集为 0。
  - 两组 record_id 均列在 `preflight.json`。
- **硬件与环境**：1× NVIDIA B200（Slurm job 48907334，节点 e4-1）；conda env `later-b200`（torch 2.8.0+cu128，transformers 5.12.1）。
- **运行模式**：fp32 权重 + bf16 autocast，`eval()` + `inference_mode()`。
- **用时**：主实验 91 s；噪声底补充实验约 60 s。

```bash
cd /scratch/hz33227/e2c/stage2/LaTER
srun --jobid=48907334 --overlap --gres=gpu:1 ~/.conda/envs/later-b200/bin/python -m later.scripts.train.p0_residual_amplification \
  --model /scratch/hz33227/e2c/stage2/runs/stage2_L8_rms_2900/checkpoint-500 \
  --data_dir /scratch/hz33227/e2c/stage2/data --out /scratch/hz33227/e2c/stage2/outputs/p0_residual_amplification
srun --jobid=48907334 --overlap --gres=gpu:1 ~/.conda/envs/later-b200/bin/python -m later.scripts.train.p0_kl_noise_floor \
  /scratch/hz33227/e2c/stage2/runs/stage2_L8_rms_2900/checkpoint-500 /scratch/hz33227/e2c/stage2/data \
  /scratch/hz33227/e2c/stage2/outputs/p0_residual_amplification 128
```

## 输出文件

- `preflight.json`：identity 检查。
- `latent_mu.pt`：每步的 μ_t，只用校准集计算。
- `p0_metrics.json`：各条件的汇总指标。
- `per_example.csv`：5 种条件 × 128 题。
- `kl_noise_floor.json`：KL 噪声底。
- `logs.txt`：运行日志。
- `P0_REPORT.md`：结论报告。
