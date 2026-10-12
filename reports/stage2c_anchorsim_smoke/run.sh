#!/bin/bash
# usage: run.sh GPU MODE
source /scratch/hz33227/e2c/stage2/outputs/stage2c_anchorsim_smoke_20261012/run_env.sh
cd $R
CUDA_VISIBLE_DEVICES=$1 $PY -m later.scripts.stage2t.anchorsim_2c --mode $2 --model $M --s2b $S2B --out $O --ckpt $CK0 > $O/$2.log 2>&1
