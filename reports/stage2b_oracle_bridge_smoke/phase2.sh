#!/bin/bash
O=/scratch/hz33227/e2c/stage2/outputs/stage2b_oracle_bridge_smoke_20261012
$O/run.sh 0 train --mode train --resume $O/checkpoints/bridge_step5.pt --stop_at 10 && $O/run.sh 0 eval_step10 --mode eval --ckpt $O/checkpoints/bridge_step10.pt
