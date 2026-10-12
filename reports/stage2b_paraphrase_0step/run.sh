#!/bin/bash
source /scratch/hz33227/e2c/stage2/outputs/stage2c_anchor_lambda01_20261012/run_env.sh
O=/scratch/hz33227/e2c/stage2/outputs/stage2b_paraphrase_0step_20261012
cd $R
CUDA_VISIBLE_DEVICES=0 $PY -m later.scripts.stage2t.paraphrase_0step --model $M --s2b $S2B --a2 $A2 --dec $DEC --init $CK0 \
  --ckpt /scratch/hz33227/e2c/stage2/outputs/stage2b_4sample_overfit_20261012/checkpoints/bridge_step20.pt --para $O/paraphrases.json --out $O > $O/run.log 2>&1
