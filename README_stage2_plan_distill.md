# OUR Stage 2 — Deterministic Latent Plan Distillation (on LaTER)

`objective_mode=plan_distill_stage2`. This is **not** LaTER's internal `train_stage=2` curriculum.
Start point: our Stage-1 explicit-planning SFT checkpoint (Qwen3-4B-E2C-SFT-only, step 800). Data: only the
Stage-1 SFT records (0 LaTER-69K samples).

```
L_total = L_execution + λa · L_alignment + λr · L_explicit
L_execution  = −log π_θ(E, A | Q, z_1..z_L)          all E + A (+ closing tag/EOS) tokens, mean over tokens
L_alignment  = (1/|H|) Σ_l mean_b ‖h_S,b^l − sg(h_T,b^l)‖₁ / (d · (sg(std_d h_T,b^l) + ε))   (fp32)
L_explicit   = −log π_θ(P, E, A | Q)                 same trainable student, explicit path
```
No teacher KL, no halting loss, no noise, no RL (asserted at start-up).

## Format (Stage-1 separators, no new tokens)
```
teacher  : prompt | <EXPLORATION>\n P \n</EXPLORATION><EXECUTION>\n                   ← h_T at last token ('>\n', id 397)
student  : prompt | <EXPLORATION>\n z_1..z_L </EXPLORATION><EXECUTION>\n | E </EXECUTION><|im_end|>
                                   ↑ seed = last token of '<EXPLORATION>\n'   ↑ h_S at '>\n'; E/A supervised after it
explicit : prompt | <EXPLORATION>\n P \n</EXPLORATION><EXECUTION>\n E </EXECUTION><|im_end|>   (whole response supervised)
```
Latent recursion = LaTER's `_build_latent_recursive_embeds` with `latent_embed_builder_no_grad=false`:
z_1 = proj(h[seed]), z_k = proj(h(z_{k−1})). Alignment layers are decoder-layer outputs 1..36 (residual stream,
before the final norm; `all`, `last4`, `last8` supported).

## Files
| file | role |
|---|---|
| `later/scripts/data/stage1_to_stage2_data.py` | Stage-1 → Stage-2 records + `stage1_to_stage2_data_audit.json` |
| `later/src/train/stage2_plan_dataset.py` | three views, labels, boundary/latent indices, collator |
| `later/scripts/data/cache_plan_boundary_hidden.py`, `later/src/train/stage2_teacher_cache.py` | frozen-teacher prefix-only boundary cache + manifest (hash-checked) |
| `later/src/train/modeling_latent.py` (+206 lines, additive) | `forward_plan_distill_stage2`, boundary capture hooks, `generate_plan_latent` |
| `later/src/train/losses.py` (+53 lines, additive) | normalized multi-layer L1, masked CE, total |
| `later/src/train/stage2_objective.py` | the three-term objective (one DDP forward) |
| `later/src/train/train_stage2.py` | trainer: DDP, fp32 master + bf16 autocast, logging, eval, save/resume |
| `later/src/train/train.py` (+2 lines) | YAML `latent_embed_builder_no_grad` is now actually passed to the config |
| `later/src/config/stage2_plan_kd_qwen3.yaml`, `later/scripts/train/run_stage2_plan_kd_qwen3.sh` | config + launcher |
| `later/src/train/test/test_stage2_plan_distill.py` | 24 contract tests (real checkpoint + data, GPU) |
| `later/scripts/train/stage2_debug_reports.py` | `batch_debug.json`, `gradient_report.json`, 20 three-view samples |
| `later/scripts/train/stage2_latent_ablation_ce.py` | forward-only latent-usage ablation |
| `later/src/eval/eval_stage2.py` | generation eval: Stage-1 explicit / Stage-2 latent / Stage-2 explicit / zero / no latent |

## Commands (env `~/.conda/envs/later-b200`, run from the repo root inside the GPU allocation)
```bash
export PYTHONPATH=$PWD
E=/scratch/hz33227/e2c/Explore-Execute-Chain; S=/scratch/hz33227/e2c/stage2
# 1. data + audit
python later/scripts/data/stage1_to_stage2_data.py --stage1_train $E/data/processed/sft/e2c-sft-train-fullfit.parquet \
  --stage1_val $E/data/processed/sft/e2c-sft-val-fullfit.parquet --eval_dir $E/data/evaluation \
  --checkpoint $E/outputs/e2c_qwen3_4b_sft/best_hf --out_dir $S/data
# 2. frozen-teacher boundary cache (≈9 min on one B200, 7.5 GB)
python later/scripts/data/cache_plan_boundary_hidden.py --teacher $E/outputs/e2c_qwen3_4b_sft/best_hf \
  --data train=$S/data/stage2_train.parquet val=$S/data/stage2_val.parquet --out $S/teacher_cache --layers all
# 3. tests (must pass before training)
python -m pytest -q later/src/train/test/test_stage2_plan_distill.py
# 4. train (auto-resumes from output_dir/latest_checkpoint.txt); overrides are key=value
NGPUS=2 bash later/scripts/train/run_stage2_plan_kd_qwen3.sh output_dir=$S/runs/my_run max_train_steps=100
# 5. evaluate
python -m later.src.eval.eval_stage2 --stage1 $E/outputs/e2c_qwen3_4b_sft/best_hf --stage2 $S/runs/my_run/checkpoint-100 \
  --val $S/data/stage2_val.parquet --math500 $E/data/evaluation/math500.parquet --out $S/eval/my_run
```
Ablations: `lambda_a=0` (w/o alignment), `lambda_r=0` (w/o explicit), `execution_ce_weight=0` (w/o execution),
`alignment_layers=last4`.

Checkpoints: `output_dir/checkpoint-N/` holds an HF `latent_qwen3` model (fp32 master weights + projector),
tokenizer, `trainer_state.pt` (optimizer, step, RNG) and `stage2_meta.json`. Load it with
`later.src.train.stage2_common.load_stage2_model(path, ...)` and generate with `model.generate_plan_latent(...)`.
