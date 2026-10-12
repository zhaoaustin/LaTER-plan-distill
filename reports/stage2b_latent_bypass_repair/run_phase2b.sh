#!/bin/bash
source /scratch/hz33227/e2c/stage2/outputs/stage2b_latent_bypass_repair_20261012/run_env.sh
$O/run.sh 1 train_A005 --mode train --init $CK0 --tag A005 --lam_anchor 0.05 --stop_at 10 &
$O/run.sh 1 train_B005 --mode train --init $CK0 --tag B005 --lam_plan 0.05 --stop_at 10 &
wait
$O/run.sh 1 train_A01 --mode train --init $CK0 --tag A01 --lam_anchor 0.1 --stop_at 10 &
for t in A001 B002; do $O/run.sh 1 diag10_$t --mode probe --ckpt $O/checkpoints/${t}_add10.pt; $O/run.sh 1 diag10_$t --mode extra --ckpt $O/checkpoints/${t}_add10.pt; done
wait
