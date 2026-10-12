#!/bin/bash
# usage: run.sh GPU LOGNAME <bridge_oracle args>
source /scratch/hz33227/e2c/stage2/outputs/stage2c_anchor_lambda01_20261012/run_env.sh
O=/scratch/hz33227/e2c/stage2/outputs/stage2b_oracle_bridge_smoke_20261012
cd $R; G=$1; LG=$2; shift 2
CUDA_VISIBLE_DEVICES=$G $PY -m later.scripts.stage2t.bridge_oracle --model $M --s2b $S2B --out $O --init $CK0 "$@" >> $O/$LG.log 2>&1
