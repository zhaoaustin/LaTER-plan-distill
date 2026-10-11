#!/usr/bin/env bash
# Final evaluation of the PlanRec pilot: step0 (Stage 1 + seed projector), A@STEP, B@STEP on 2 GPUs.
#   bash later/scripts/train/run_planrec_eval.sh [STEP=300]
set -euo pipefail
STEP="${1:-300}"
PY="${PY:-$HOME/.conda/envs/later-b200/bin}"
export PATH="$PY:$PATH" PYTHONPATH="$(pwd):${PYTHONPATH:-}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
ROOT=/scratch/hz33227/e2c/stage2/outputs/stage2_planrec_two_loss_pilot
S1=/scratch/hz33227/e2c/Explore-Execute-Chain/outputs/e2c_qwen3_4b_sft/best_hf
COMMON="--stage1 $S1 --val /scratch/hz33227/e2c/stage2/data/stage2_val.parquet --n 200 --gen_data /scratch/hz33227/e2c/stage2/data/clean_math_eval.parquet --gen_n 50 --out $ROOT/evaluation"
mkdir -p $ROOT/evaluation
CUDA_VISIBLE_DEVICES=0 python later/scripts/train/eval_planrec_pilot.py --name B_step$STEP --model $ROOT/run_B_execution_plus_planrec/checkpoint-$STEP $COMMON > $ROOT/evaluation/eval_B.log 2>&1 &
( CUDA_VISIBLE_DEVICES=1 python later/scripts/train/eval_planrec_pilot.py --name A_step$STEP --model $ROOT/run_A_execution_only/checkpoint-$STEP $COMMON > $ROOT/evaluation/eval_A.log 2>&1
  CUDA_VISIBLE_DEVICES=1 python later/scripts/train/eval_planrec_pilot.py --name step0 --model $S1 --projector $ROOT/run_B_execution_plus_planrec/projector_step0.pt $COMMON > $ROOT/evaluation/eval_step0.log 2>&1 ) &
wait
echo EVAL_ALL_DONE
