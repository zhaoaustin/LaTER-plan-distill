#!/bin/bash
O=/scratch/hz33227/e2c/stage2/outputs/stage2b_oracle_bridge_smoke_20261012
$O/run.sh 0 audit --mode audit
grep -q '"PASS": true' $O/audit.json || { echo AUDIT_FAIL; exit 1; }
$O/run.sh 0 eval_step0 --mode eval &
( $O/run.sh 1 train --mode train --stop_at 5 && $O/run.sh 1 eval_step5 --mode eval --ckpt $O/checkpoints/bridge_step5.pt ) &
wait
