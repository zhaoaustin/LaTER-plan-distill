#!/bin/bash
source /scratch/hz33227/e2c/stage2/outputs/stage2c_anchor_lambda01_20261012/run_env.sh
O=/scratch/hz33227/e2c/stage2/outputs/stage2b_teacher_vs_student_0step_20261012
cd $R
CUDA_VISIBLE_DEVICES=0 $PY -m later.scripts.stage2t.teacher_vs_student_0step --model $M --s2b $S2B --a2 $A2 --dec $DEC --ckpt $CK0 \
  --bridge_pairs /scratch/hz33227/e2c/stage2/outputs/mini_bridge_pilot/heldout_pairs.jsonl --out $O > $O/run.log 2>&1
