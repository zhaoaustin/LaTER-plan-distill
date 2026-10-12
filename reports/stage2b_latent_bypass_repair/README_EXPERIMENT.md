# Stage 2B Latent Bypass Repair — 运行说明

所有路径都取自本机实际存在的文件。环境变量定义在 `run_env.sh`：PY、M、S2B、A2、O、CK0、R。

## 环境
- Slurm 作业 48907334（节点 e4-1，2× NVIDIA B200）。
- conda 环境 `~/.conda/envs/later-b200`（torch 2.8.0+cu128，transformers 5.12.1，peft 0.18.1），bf16 autocast，离线 HF。
- 代码：仓库 `$R=/scratch/hz33227/e2c/stage2/LaTER`，分支 `exp/stage2b-latent-bypass-repair`，基于 15ed0ce。
  - `later/scripts/stage2t/contrastive_2b.py`：audit / train / probe / extra / acc 五种模式。
  - `later/scripts/stage2t/test_contrastive.py`：单元测试和 smoke test。
  - `later/scripts/stage2t/contrastive_report.py`：汇总表格并画图。
  - 原来的 `joint_2b.py`、`three_step.py` 没有改动，生成评估直接复用 `three_step.py --mode eval --gen`。
- 起点：`$CK0` = `stage2b_joint_two_loss_20261012/checkpoints/joint_step50.pt`，sha256 前 16 位见 `ck0.sha`。其中包含 projector、LoRA 和 AdamW 状态，没有被覆盖。

## 进入作业
```bash
srun --jobid=48907334 --overlap --gres=gpu:2 <脚本>
```
脚本内部用 `CUDA_VISIBLE_DEVICES=0/1` 选卡。

## 复现顺序
```bash
O=/scratch/hz33227/e2c/stage2/outputs/stage2b_latent_bypass_repair_20261012
srun --jobid=48907334 --overlap --gres=gpu:2 $O/run_phase1.sh     # Gate (audit_gate.json) + 单测 (test_contrastive.json)
srun --jobid=48907334 --overlap --gres=gpu:2 $O/run_phase2.sh     # D0 50 步；A λ=0.01、B λ=0.02 各 10 步
srun --jobid=48907334 --overlap --gres=gpu:2 $O/run_phase2b.sh    # A λ=0.05/0.1、B λ=0.05 各 10 步 + 10 步 probe/extra
srun --jobid=48907334 --overlap --gres=gpu:2 $O/run_phase2c.sh    # 其余 10/25 步 probe；D0 第 50 步完整评估
srun --jobid=48907334 --overlap --gres=gpu:2 $O/run_eval.sh 0 $O/start/start50.pt three_step   # 起点完整评估
source $O/run_env.sh; cd $R && $PY -m later.scripts.stage2t.contrastive_report $O              # metrics.csv / eval.json / cases.jsonl / fig1-4
```

单独运行某一项时用 `run.sh GPU 日志名 <contrastive_2b 参数>`：
```bash
$O/run.sh 0 train_X --mode train --init $CK0 --tag X --lam_anchor 0.01 --stop_at 10      # 从起点训练
$O/run.sh 0 train_X --mode train --resume $O/checkpoints/X_add10.pt --tag X --lam_anchor 0.01 --stop_at 50   # 续跑（数据序与优化器状态都接上）
$O/run_eval.sh 0 $O/checkpoints/X_add50.pt three_step                                    # probe + extra + 128 题准确率 + 生成评估
```

## 输出
- `audit_gate.json`：G0/G1/G2 检查。
- `test_contrastive.json`：16 项硬性单测。
- `train_<tag>.jsonl`、`metrics.csv`：逐步记录 loss 和梯度。
- `evaluations/`：
  - `probe_*`：48 道未见题上的 TF/FR cos；
  - `extra_*`：JSD、ΔCE 和同题 2×2 似然；
  - `acc_*`：128 道独立未见题的准确率；
  - `summary_*`、`rows_*`、`generations_*`：同题 A/B 生成。
- `eval.json`：以上各项的汇总。
- `cases.jsonl`：按题记录生成结果。
- `fig1-4*.png`：四张图。
- `FINAL_CONTRASTIVE_COMPARISON.md`：最终报告。
- `checkpoints/`：每个分支的 `<tag>_add{10,25,50}.pt`。
