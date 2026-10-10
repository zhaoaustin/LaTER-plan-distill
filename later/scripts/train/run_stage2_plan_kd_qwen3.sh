#!/usr/bin/env bash
# OUR Stage 2 plan distillation. Usage (inside the GPU allocation):
#   NGPUS=2 CONFIG=later/src/config/stage2_plan_kd_qwen3.yaml bash later/scripts/train/run_stage2_plan_kd_qwen3.sh [key=value ...]
set -euo pipefail
cd "$(dirname "$0")/../../.."
NGPUS="${NGPUS:-2}"
CONFIG="${CONFIG:-later/src/config/stage2_plan_kd_qwen3.yaml}"
PY="${PY:-$HOME/.conda/envs/later-b200/bin}"
export PATH="$PY:$PATH" PYTHONPATH="$(pwd):${PYTHONPATH:-}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
torchrun --standalone --nproc-per-node="$NGPUS" -m later.src.train.train_stage2 --config "$CONFIG" --override "$@"
