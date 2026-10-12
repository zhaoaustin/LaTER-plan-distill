#!/bin/bash
# usage: run.sh GPU LOGNAME <contrastive_2b args...>
source /scratch/hz33227/e2c/stage2/outputs/stage2b_latent_bypass_repair_20261012/run_env.sh
cd $R
G=$1; LG=$2; shift 2
CUDA_VISIBLE_DEVICES=$G $PY -m later.scripts.stage2t.contrastive_2b --model $M --s2b $S2B --out $O "$@" >> $O/logs/$LG.log 2>&1
