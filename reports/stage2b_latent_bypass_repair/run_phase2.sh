#!/bin/bash
source /scratch/hz33227/e2c/stage2/outputs/stage2b_latent_bypass_repair_20261012/run_env.sh
mkdir -p $O/logs
$O/run.sh 0 train_D0 --mode train --init $CK0 --tag D0 --stop_at 50 &
( $O/run.sh 1 train_A001 --mode train --init $CK0 --tag A001 --lam_anchor 0.01 --stop_at 10 ;
  $O/run.sh 1 train_B002 --mode train --init $CK0 --tag B002 --lam_plan 0.02 --stop_at 10 ) &
wait
