#!/bin/bash
source /scratch/hz33227/e2c/stage2/outputs/stage2b_latent_bypass_repair_20261012/run_env.sh
cd $R
CUDA_VISIBLE_DEVICES=0 $PY -m later.scripts.stage2t.contrastive_2b --mode audit --model $M --s2b $S2B --out $O --ckpt $CK0 > $O/audit.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY -m later.scripts.stage2t.test_contrastive --model $M --s2b $S2B --ckpt $CK0 --out $O > $O/test.log 2>&1
wait
