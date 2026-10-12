#!/bin/bash
# usage: run_eval.sh GPU CKPT [three_step]  -> probe + extra + acc (+ three_step eval --gen)
source /scratch/hz33227/e2c/stage2/outputs/stage2b_latent_bypass_repair_20261012/run_env.sh
G=$1; CK=$2; N=$(basename $CK .pt); mkdir -p $O/logs
$O/run.sh $G eval_$N --mode probe --ckpt $CK
$O/run.sh $G eval_$N --mode extra --ckpt $CK
$O/run.sh $G eval_$N --mode acc --ckpt $CK
if [ "$3" == "three_step" ]; then cd $R; CUDA_VISIBLE_DEVICES=$G $PY -m later.scripts.stage2t.three_step --mode eval --gen --model $M --s2b $S2B --out $O --ckpt $CK >> $O/logs/eval3_$N.log 2>&1; fi
