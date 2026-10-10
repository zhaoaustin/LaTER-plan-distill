# Stage 2 — Deterministic Latent Plan Distillation: implementation & smoke-validation report

Date 2026-10-10 · 2 × NVIDIA B200 · LaTER @ 6df72f7 + commit d508fc6 (this work)

**Status: the implementation is complete and verified (25/25 contract tests, gradients, boundary, masks,
train/infer equivalence, save/resume). After a 100-step smoke run the latents are not yet used for
execution. The Stage-3 gate "latent actually used" is NOT met.** Details in §5–§7.

## 1. What was built
Three-term objective, exactly as specified (no KL, no halting loss, no noise, no RL; asserted at start-up):
`L = L_execution + λa·L_alignment + λr·L_explicit`, with λa = 0.1 and λr = 0.3 as smoke-only placeholders.
- **L_execution**: CE over every E + A token (plus the closing `</EXECUTION><|im_end|>`) of the latent path,
  token mean.
- **L_alignment**: normalized multi-layer L1, computed in fp32. All 36 decoder-layer outputs (residual
  stream, before the final norm); teacher std over d (correction = 0); ε = 1e-6; frozen prefix-only teacher
  cache; stop-grad.
- **L_explicit**: CE over all P + E + A tokens of the **same** trainable student on the explicit path.
- The latent recursion is LaTER's `_build_latent_recursive_embeds` with `latent_embed_builder_no_grad=false`.
  `train.py` now actually passes this flag from YAML; before, the YAML value was silently ignored.
- Boundary = the Stage-1 separator's last token `>\n` (id 397) of `</EXPLORATION><EXECUTION>\n`. It is
  identical in the teacher and the student (9-token suffix). No new tokens are introduced.

## 2. Data (Stage-1 SFT only; 0 LaTER-69K samples)
- 49,684 / 2,611 Stage-1 rows → **46,510 / 2,446 accepted**.
- Rejected: 363 + 14 with no parsable answer; 2,542 + 127 whose plan states the final answer; 23 whose
  plan contains `\boxed`; **320 overlapping AIME24/25, MATH-500, OlympiadBench or AMC23**; 7 val rows whose
  question also appears in train.
- Every accepted record reassembles byte-exactly into its Stage-1 target.
- Note: Stage 1 itself was trained on the 320 benchmark-overlapping records (see the preflight report).

## 3. Tests (`later/src/train/test/test_stage2_plan_distill.py`, real checkpoint + data): 25 passed
The suite covers:
- provenance, lossless parsing, group split and eval overlap;
- the three views; the exact boundary, including string-level prefix equivalence;
- E/A and P/E/A masks; normalized L1 against a float64 reference;
- layer-index consistency against `output_hidden_states`, and pre-final-norm semantics;
- no future leak (prefix-only and full-causal);
- projector / backbone / early-latent gradients for each loss alone and for the full loss;
- `no_grad=True` gives a projector gradient of exactly 0;
- frozen teacher and exact total loss; L = 2, 4, 8; gradient checkpointing;
- train/infer equivalence (identical argmax, max logit diff < 0.05);
- save/resume; bf16 smoke on the longest record (peak 50 GiB);
- teacher cache vs online computation; projector output RMS.

## 4. Gradient report (L = 8, bf16, gradients at initialization)
| loss | projector ‖g‖ | early latent ‖g‖ | backbone ‖g‖ |
|---|---|---|---|
| execution only | 9.40 | 0.109 | 44.5 |
| alignment only | 23.2 | 0.142 | 92.0 |
| explicit only | **0** | 0 | 2.75 |
| full | 7.48 | 0.116 | 41.6 |

The same holds with gradient checkpointing on; L = 2 and L = 4 also give non-zero early-latent gradients.

## 5. Bug found during smoke: latent scale
- In the first 100-step run, the projector outputs had norm **68–87** while token embeddings have norm
  **1.01**.
- RMSNorm divides the gradient reaching such a vector by its norm. As a result, the early-latent gradient
  fell to about 2e-6, and the projector gradient to about 0.003.
- Fix: optional `latent_projector_output_rms: token_embedding`, which rescales projector outputs to the
  token-embedding RMS (0.0220 per dim). It is off by default, so the LaTER baseline is unchanged.
- After the fix, latent norm is 1.11 and the early-latent gradient is about 700× larger (0.017–0.096 at
  step 16).

## 6. Smoke training (fixed config; DDP over 2 GPUs; global batch 16; 100 steps ≈ 0.03 epoch; 5.4 s/step; peak ≈ 115 GiB/GPU)
| val (100 records) | step 25 | 50 | 75 | 100 |
|---|---|---|---|---|
| L_execution | 0.248 | 0.245 | 0.243 | 0.243 |
| L_alignment | 0.177 | 0.119 | 0.102 | 0.099 |
| L_explicit | 0.195 | 0.193 | 0.191 | 0.191 |

Earlier checks:
- tiny-set overfit (8 records): L_exec 0.25 → 0.011, L_explicit 0.14 → 0.015 in 10 steps;
- resume from step 5 continued from the same loss.

## 7. Validation
### 7.1 Forward-only latent ablation (200 val records, teacher forcing)
| model | gold plan | learned latent | zero latent | no latent |
|---|---|---|---|---|
| Execution CE, Stage 1 (random projector) | 0.147 | 0.277 | 0.277 | 0.278 |
| Execution CE, Stage 2 step 100 | 0.153 | **0.2348** | **0.2349** | **0.2349** |
| Alignment loss, Stage 2 step 100 | – | **0.099** | 0.111 | 0.117 |

- Plan information recovered by the latents in execution CE: **0.16%**.
- The latents are starting to shape the boundary state: alignment with learned latents is 11–15% lower
  than with zero or no latents. That information does not reach execution yet.

### 7.2 Generation (greedy, ≤ 3,072 new tokens; 100 val + 100 MATH-500)
| condition | val agreement* | MATH-500 acc | MATH-500 complete CoT | MATH-500 mean tokens |
|---|---|---|---|---|
| Stage 1 explicit (P → E) | 0.45 | 0.75 | 0.95 | 760 |
| Stage 1 empty plan | 0.15 | 0.29 | 0.36 | 258 |
| **Stage 2 latent (main)** | 0.44 | **0.77** | 0.94 | 654 |
| Stage 2 explicit (anti-forgetting) | 0.42 | 0.76 | 0.97 | 738 |
| Stage 2 zero latent | 0.44 | 0.77 | 0.96 | 591 |
| Stage 2 no latent | 0.42 | 0.81 | 0.97 | 597 |

\*Agreement with the Stage-1 target answer, which is unverified (no gold answers in the data).
On 100 problems one problem is 1 point, so differences of a few points are within noise.

### 7.3 Interpretation
- **Stage-2 latent generation works end to end.** Outputs are complete, closed executions with valid
  answers (94% complete CoT on MATH-500), and no plan tags are regenerated.
- **Explicit ability is preserved:** Stage-2 explicit 0.76 vs Stage-1 explicit 0.75.
- **The latents are not used:** learned, zero and no-latent give the same execution CE, and generation is
  equal within noise.
- Stage 1 collapses without a plan (0.29). Stage 2 learned to execute without one within 100 steps, so the
  backbone found a shortcut. In this data each question has exactly one plan and one execution, so E can be
  predicted from Q directly. The boundary alignment can likewise be met largely by computation over Q and the
  separator tokens.
- This is a method-level risk, not an implementation defect.

## 8. Known issues / risks
1. **Latent shortcut**, as above. 100 steps cannot tell whether latents become useful with longer
   training.
2. Stage-1 checkpoint saw 320 benchmark-overlapping records, so AIME and MATH-500 numbers for these
   checkpoints are optimistic. The 20-problem Stage-1 acceptance set was 13-gram clean.
3. On harder val problems, 12–15% of outputs from every model do not close within 3,072 tokens, i.e. they
   loop.
4. Upstream LaTER `train.py` imports a missing module (`trainer_latent_cp`), so the original LaTER
   training entry does not run at 6df72f7. Not patched.
5. λa, λr, learning rates (backbone 1e-5, projector 1e-4) and L = 8 are untuned placeholders.
6. Teacher and student boundary RoPE positions differ (≈ 289 vs ≈ prompt + 23). This is logged and not
   adjusted.

## 9. Suggested next actions (not started; need GPU time approval)
1. Longer run with the latent-usage probe (forward ablation every 250 steps), e.g. one epoch:
   2,900 steps ≈ 4.5 h on 2 × B200.
2. Make the latents necessary instead of optional, for example:
   - larger λa, or alignment-first warm-up;
   - several sampled plans per question, so E is not a function of Q alone;
   - latent dropout on the question tokens after the latent block (CODI-style experiments).
   Each needs a separate config and must be labelled as a deviation.
3. Sweep λa ∈ {0.1, 0.3, 1.0} and λr ∈ {0.1, 0.3}; compare `all` vs `last4`.
