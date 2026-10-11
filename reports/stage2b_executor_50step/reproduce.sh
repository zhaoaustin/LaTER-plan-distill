#!/bin/bash
set -euo pipefail
cd /scratch/hz33227/e2c/stage2/LaTER   # branch exp/stage2b-executor-50step
PY=~/.conda/envs/later-b200/bin/python; M=/scratch/hz33227/e2c/Explore-Execute-Chain/outputs/e2c_qwen3_4b_sft/best_hf
A2=/scratch/hz33227/e2c/stage2/outputs/stage2a_external_plan_ae; O=/scratch/hz33227/e2c/stage2/outputs/stage2b_executor_50step
DEC=/home/hz33227/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
srun --jobid=48907334 --overlap --gres=gpu:1 $PY -m later.scripts.stage2b.build_cache --a2 $A2 --dec $DEC \
  --rules /scratch/hz33227/e2c/stage2/outputs/stage2_fast_plan_specific_diagnostics/span_rules_frozen.json --data_dir /scratch/hz33227/e2c/stage2/data \
  --bridge /scratch/hz33227/e2c/stage2/outputs/bridge_input_abcd --exec_tok $M --out $O/data
srun --jobid=48907334 --overlap --gres=gpu:1 $PY -m later.scripts.stage2b.executor_2b --mode gates --model $M --data $O/data --out $O
srun --jobid=48907334 --overlap --gres=gpu:2 ~/.conda/envs/later-b200/bin/torchrun --nproc_per_node 2 -m later.scripts.stage2b.executor_2b --mode train --steps 50 --model $M --data $O/data --out $O
for s in 0 25 50; do G=""; [ $s != 25 ] && G="--gen"; srun --jobid=48907334 --overlap --gres=gpu:1 $PY -m later.scripts.stage2b.executor_2b --mode eval $G --ckpt $O/checkpoints/step$s.pt --model $M --data $O/data --out $O; done
