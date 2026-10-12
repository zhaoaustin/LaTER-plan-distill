# REPRODUCE

```bash
source /scratch/hz33227/e2c/stage2/outputs/stage2_three_step_autonomous_20261012/run_env.sh   # PYTHONPATH, PY, M, S2B, A2, O, DEC
cd $O
# Gate 1（14 项单测）
$PY -m later.scripts.stage2t.three_step --mode gates --model $M --s2b $S2B --out $O
# Gate 1.9 Tiny overfit（10 步）
$PY -m later.scripts.stage2t.three_step --mode train --tiny --steps 10 --choice --lam_choice 1.0 --tag tiny --model $M --s2b $S2B --out $O
# Stage 2B KD warm-up（40 步）
$PY -m later.scripts.stage2t.three_step --mode train --phase 2b --steps 40 --n_single 16 --n_pair 0 --lam_exec 0 --eval_at 10,20,40 --tag 2b --model $M --s2b $S2B --out $O
# 评估（只算分数，加 2A reader）
for s in 0 20 40; do $PY -m later.scripts.stage2t.three_step --mode eval --ckpt $O/checkpoints/2b_step$s.pt --a2 $A2 --dec $DEC --model $M --s2b $S2B --out $O; done
# 只读诊断
$PY $O/teacher_audit.py $A2 $S2B $O                     # 2A reader 对 28 对 A/B Plan 的自由生成
$PY $O/q2z_probe.py $M $S2B $O                          # 用冻结 Stage-1 的 Q 表征做 ridge / 1-NN，预测 Z_T
$PY -m later.scripts.stage2t.three_step --mode audit --ckpt $O/checkpoints/2b_step40.pt --n_single 8 --n_pair 2 --model $M --s2b $S2B --out $O
# 2C（本轮因 KD_COLLAPSE 没有运行；如获批准，命令如下，两个分支同时跑、各占一张卡）
# CUDA_VISIBLE_DEVICES=0 $PY -m later.scripts.stage2t.three_step --mode train --phase 2c --init $O/checkpoints/2b_step40.pt --steps 55 --n_single 8 --n_pair 2 --eval_at 27 --tag 2c_control ...
# CUDA_VISIBLE_DEVICES=1 $PY -m later.scripts.stage2t.three_step --mode train --phase 2c --choice --lam_choice 0.1758 --init $O/checkpoints/2b_step40.pt --steps 55 --n_single 8 --n_pair 2 --eval_at 27 --tag 2c_choice ...
```
注意：本机用 `srun --jobid=<hold> --overlap --gres=gpu:2` 进入作业，再用 CUDA_VISIBLE_DEVICES 指定卡。如果用 `--gres=gpu:1`，所有进程都会落到 GPU0。
