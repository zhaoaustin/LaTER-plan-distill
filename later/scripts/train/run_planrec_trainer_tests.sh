#!/usr/bin/env bash
# Trainer-level gates for the PlanRec pilot (2 GPUs): DDP + save/resume equivalence, then tiny overfit.
#   bash later/scripts/train/run_planrec_trainer_tests.sh <tests_dir>
set -euo pipefail
OUT="$1"
PY="${PY:-$HOME/.conda/envs/later-b200/bin}"
export PATH="$PY:$PATH" PYTHONPATH="$(pwd):${PYTHONPATH:-}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
CFG=later/src/config/stage2_planrec_two_loss.yaml
run() { torchrun --standalone --nproc-per-node=2 -m later.src.train.train_stage2_planrec --config "$CFG" --override "$@"; }
COMMON="wandb=false train_limit=64 probe_steps=[] save_weights_only_steps=[] save_steps=[]"
# (1) 4 uninterrupted steps
run $COMMON output_dir=$OUT/ddp_resume/full max_train_steps=4 > $OUT/ddp_resume_full.log 2>&1
# (2) 2 steps, then resume from its own checkpoint-2 to step 4
run $COMMON output_dir=$OUT/ddp_resume/split max_train_steps=2 > $OUT/ddp_resume_split_a.log 2>&1
run $COMMON output_dir=$OUT/ddp_resume/split max_train_steps=4 resume=true > $OUT/ddp_resume_split_b.log 2>&1
echo DDP_RESUME_RUNS_DONE
# (3) tiny overfit: 16 train records, global batch 16 (every step sees all 16), 40 steps; probe on the same records
run wandb=false train_limit=16 max_train_steps=40 lr_schedule_max_steps=40 probe_steps=[0,20,40] probe_n=16 \
    probe_data=/scratch/hz33227/e2c/stage2/data/stage2_train.parquet save_steps=[] save_weights_only_steps=[] \
    output_dir=$OUT/tiny_overfit > $OUT/tiny_overfit.log 2>&1
echo TINY_OVERFIT_DONE
