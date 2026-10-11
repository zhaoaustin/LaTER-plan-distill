# Stage 2 two-loss PlanRec pilot: how to reproduce

Code: https://github.com/zhaoaustin/LaTER-plan-distill, branch `exp/stage2-planrec-two-loss`
(training ran at commit `727c016`; the launcher was committed right after as `f6bdc61`). Base for the diff:
`2884f31` (execution-KL pilot). `changes.diff` = `git diff 2884f31..HEAD`.

Environment: 2x NVIDIA B200 (Slurm `iai_B200_p`), conda env `later-b200` (torch 2.8, transformers as recorded in
`run_*/run_config_resolved.json`), fp32 master weights + bf16 autocast, SDPA attention, DDP.

Inputs:
- Stage-1 model `zhaohq/Qwen3-4B-E2C-SFT-only` @ `a6b0e2fe008f14b00bf7f3a29a214e96212f7251` (step 800). Local copy
  `/scratch/hz33227/e2c/Explore-Execute-Chain/outputs/e2c_qwen3_4b_sft/best_hf`; both safetensors shards match the
  HF revision's LFS sha256 (see `dataset_manifest.json`).
- Data: `stage2_{train,val}.parquet` (46,510 / 2,446), converted from the E2C Stage-1 SFT split only
  (no LaTER-69K, no new teacher data). Hashes and audits in `dataset_manifest.json`.

Commands (from the repo root):
```bash
export PYTHONPATH=$(pwd)
# 0. tests (GPU): 10 PlanRec contract tests + the 30 three-loss tests
python -m pytest -q later/src/train/test/test_stage2_planrec.py later/src/train/test/test_stage2_plan_distill.py
# 1. trainer gates: DDP + save/resume equivalence, tiny overfit (16 records, 40 steps)
bash later/scripts/train/run_planrec_trainer_tests.sh <out>/tests
# 2. data manifest
python later/scripts/data/make_planrec_manifest.py --stage1 <stage1> --data_dir <data> --out <out>/dataset_manifest.json
# 3. pilot: run A (lambda_p = 0) then run B (lambda_p = 0.5), 300 steps each, global batch 16
bash later/scripts/train/run_planrec_pilot.sh
# 4. final evaluation (step 0, A@300, B@300) and aggregation
bash later/scripts/train/run_planrec_eval.sh 300
python later/scripts/train/planrec_pilot_report.py --root <out>
```

Configs: `later/src/config/stage2_planrec_two_loss.yaml` (B) and `stage2_planrec_A_execution_only.yaml` (A); they
differ only in `lambda_p`, `output_dir` and W&B names. Resolved copies: `resolved_config.yaml`.

Layout:
- `tests/`: pytest logs, DDP/resume logs and batch files, tiny-overfit log, probes and plan reconstructions. The
  test checkpoints were deleted after the checks (45 GB each).
- `run_A_execution_only/`, `run_B_execution_plus_planrec/`:
  - `checkpoint-300`: model + projector + optimizer + RNG + sampler state (resumable);
  - `checkpoint-100`: weights only;
  - `projector_step0.pt`; train/probe logs; per-step batch record ids per rank; W&B id.
- `evaluation/`: `metrics.json`, `per_example.csv`, `samples.jsonl`, `report.md`.
