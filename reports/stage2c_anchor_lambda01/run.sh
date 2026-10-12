#!/bin/bash
source /scratch/hz33227/e2c/stage2/outputs/stage2c_anchor_lambda01_20261012/run_env.sh
cd $R
CUDA_VISIBLE_DEVICES=0 $PY -m later.scripts.stage2t.anchorsim_2c --mode train --model $M --s2b $S2B --out $O --ckpt $CK0 --lam_anchor 0.1 --stop_if_correct_le 2 > $O/train.log 2>&1
