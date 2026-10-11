#!/usr/bin/env bash
# Two-loss PlanRec pilot: run A (execution only, lambda_p=0) then run B (execution + 0.5 PlanRec), same init,
# seed, data order, steps and global batch (2 GPUs x micro 1 x accum 8 = 16).
#   bash later/scripts/train/run_planrec_pilot.sh [extra key=value overrides for both runs]
set -euo pipefail
PY="${PY:-$HOME/.conda/envs/later-b200/bin}"
export PATH="$PY:$PATH" PYTHONPATH="$(pwd):${PYTHONPATH:-}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
ROOT=/scratch/hz33227/e2c/stage2/outputs/stage2_planrec_two_loss_pilot
mkdir -p $ROOT/run_A_execution_only $ROOT/run_B_execution_plus_planrec
torchrun --standalone --nproc-per-node=2 -m later.src.train.train_stage2_planrec \
  --config later/src/config/stage2_planrec_A_execution_only.yaml --override "$@" > $ROOT/run_A_execution_only/train.log 2>&1
echo RUN_A_DONE
torchrun --standalone --nproc-per-node=2 -m later.src.train.train_stage2_planrec \
  --config later/src/config/stage2_planrec_two_loss.yaml --override "$@" > $ROOT/run_B_execution_plus_planrec/train.log 2>&1
echo RUN_B_DONE
