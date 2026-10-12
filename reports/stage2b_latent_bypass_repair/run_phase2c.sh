#!/bin/bash
source /scratch/hz33227/e2c/stage2/outputs/stage2b_latent_bypass_repair_20261012/run_env.sh
for t in A005 A01 B005 D0; do $O/run.sh 1 diag10_$t --mode probe --ckpt $O/checkpoints/${t}_add10.pt; $O/run.sh 1 diag10_$t --mode extra --ckpt $O/checkpoints/${t}_add10.pt; done
$O/run.sh 1 diag25_D0 --mode probe --ckpt $O/checkpoints/D0_add25.pt
until [ -f $O/checkpoints/D0_add50.pt ]; do sleep 15; done; sleep 20
$O/run_eval.sh 1 $O/checkpoints/D0_add50.pt three_step
