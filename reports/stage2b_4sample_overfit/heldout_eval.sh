#!/bin/bash
source /scratch/hz33227/e2c/stage2/outputs/stage2c_anchor_lambda01_20261012/run_env.sh
O=/scratch/hz33227/e2c/stage2/outputs/stage2b_4sample_overfit_20261012
mkdir -p $O/heldout_eval_step20; cd $R
CUDA_VISIBLE_DEVICES=0 $PY -m later.scripts.stage2t.bridge_oracle --model $M --s2b $S2B --out $O/heldout_eval_step20 --init $CK0 --mode eval --ckpt $O/checkpoints/bridge_step20.pt > $O/heldout_eval_step20/eval.log 2>&1
