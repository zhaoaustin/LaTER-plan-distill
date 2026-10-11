# Stage 2 Plan Reconstruction: diagnostic report (2026-10-11)

Forward-only diagnostics on the existing pilot checkpoints. Nothing was trained and no checkpoint was modified.

## Verdicts
| test | verdict |
|---|---|
| Test 1: explicit 0/50 | **EXPLICIT_FORGETTING_SUPPORTED**: format forgetting in A and B; additional math loss in B |
| Test 2: plan-reconstruction conditionality | **GENERIC_SOFT_PROMPT** (B; implementation checks passed) |
| Test 3: probe | **NOT_EVALUATED**: no reliable strategy labels exist |

## Provenance (details in `asset_manifest.json`)
- **Code:** branch `exp/stage2-planrec-two-loss`; diagnostics started at `3a52a8d`; pilot trained at `727c016`.
  The diagnostic scripts are in `changes.diff`.
- **Stage 1:** `zhaohq/Qwen3-4B-E2C-SFT-only` @ `a6b0e2fe008f14b00bf7f3a29a214e96212f7251`. Local `best_hf`
  safetensors sha256 equal the HF LFS hashes (5437b203…, 1035e45e…).
- **A / B:** `outputs/stage2_planrec_two_loss_pilot/run_{A_execution_only,B_execution_plus_planrec}/checkpoint-300`:
  model_type latent_qwen3, L = 8, projector output RMS 0.02195 (= token-embedding RMS).
- **Step 0:** Stage 1 + the saved seed-1234 projector. For explicit mode the projector is unused, so step 0 = Stage 1.
- **Tokenizer and template:** tokenizer sha aeb13307…, chat template sha a55ee1b1…, identical across all models.
- **Data:**
  - `stage2_val.parquet` sha 6b63ba49298b…. Test 2 uses the first 128 records.
  - `clean_math_eval.parquet`: Test 1, 50 problems, MATH-500, 13-gram clean against the full E²C SFT source;
    paraphrases are not detected.
- **Denominators:** Test 1 uses n = 50 per run. Test 2 uses n = 128 records with 23,086 plan-reconstruction target
  tokens and 118,199 execution target tokens.

## Q1. Is explicit mode really degraded? Yes, mostly the format; math only in B
- **Stage 1 is normal everywhere.** Format valid 0.90–0.96 and math 0.70–0.76 under native HF + native SFT protocol,
  wrapper + native protocol, and wrapper + Stage-2 loop.
  - Native vs wrapper logits: top-1 agrees 50/50, KL 1e-6.
  - The pilot's step-0 result (0.72) is reproduced exactly.
  - So there is no interface bug.
- **A and B: format valid 0/50 under both loading paths and both protocols.** Mechanism: the first token is no
  longer `<` (`<EXPLORATION>`).
  - A writes execution-style steps directly.
  - B writes an untagged plan, often with an early `Answer: $\boxed{}$` line (31/50).
- **Format-free math (last `\boxed{}`):**

  | | Stage 1 | A | B |
  |---|---|---|---|
  | math accuracy | 0.76 | 0.74–0.76 | 0.56–0.58 |
  | paired vs Stage 1 (exact McNemar) | — | p = 1.0 | p = 0.031 |
  | outputs hitting 3,072 tokens | 4% | 8–16% | 44–46% (87% repeated lines) |

  B vs A: p = 0.035.
- **Breakdown:**
  - **Format collapse:** A and B. Neither trains the explicit path; the pilot removed L_explicit by design.
  - **Truncation / repetition:** mainly B.
  - **Math errors beyond Stage 1:** B only.
  - "accuracy = 0" in the pilot report was a format-parser result, not 0% math.

## Q2. Does plan reconstruction depend on the correct latent? No (B)
Same 128 records, same P, mask and targets; only Z changes. CE(cond) − CE(learned), 95% paired bootstrap CI:

| | vs shuffled (3 derangements, mean) | vs mean latent |
|---|---|---|
| plan reconstruction, all tokens (CE learned 0.70696) | +0.00020 [−0.00004, +0.00044] | −0.00010 [−0.00037, +0.00019] |
| first 16 plan tokens | +0.00018 [−0.00037, +0.00071] | −0.00028 [−0.00083, +0.00023] |
| first 32 plan tokens | −0.00010 [−0.00070, +0.00048] | −0.00016 [−0.00086, +0.00050] |
| non-template content tokens (83.6%) | +0.00025 [−0.00003, +0.00054] | −0.00006 [−0.00038, +0.00027] |
| first 32 content tokens | +0.00015 [−0.00062, +0.00094] | +0.00025 [−0.00053, +0.00107] |
| execution CE (learned 0.23105) | +0.00007 [−0.00001, +0.00015] | +0.00003 [−0.00006, +0.00011] |

- **Every CI includes 0, including at the plan start.**
  - Decoder KL(learned‖shuffled) over the first 16 plan positions: 1.3e-4 nats.
  - First plan token without any prefix: 7e-8 nats.
  - Cross-question latent cosine 0.984; between-question variance 0.5–2.5% of latent energy.
- **Free reconstruction (20 records):**
  - 2 distinct outputs with learned latents, 2 shuffled, 1 mean;
  - all are the same unrelated "total number of students… grades" plan;
  - own-plan identification 1/20 in every condition.
- **Conclusion:** the plan-reconstruction branch learned a language-model prior over plan text that ignores Z. Under
  teacher forcing, P<t carries the rest. This contrasts with the tiny overfit (16 records), where Z did become
  partly discriminative (7/16). That shows the branch can carry information through Z, but at pilot scale and budget
  the generic solution won.
- **Implementation is valid on these checkpoints:**
  - future-token invariance holds (max logit change 0.0 before the edit);
  - Z is invariant to E/A edits (same hash);
  - swapped Z reaches both branches (the hash of the inserted tensor matches; outputs change);
  - the context contains no question tokens.
- **Reconstruction vs execution:** neither use of Z depends on the correct question. Execution is also flat, so no
  plan information reaches the executor through Z either.

## Q3. Readable information in Z: not evaluated
There are no strategy labels in the data, and inventing them is not allowed. Read-only geometry (above) shows Z is
close to a cross-question constant; this says nothing about plan semantics either way.

## Root causes, ranked by evidence
1. **Z collapses to a near-constant soft prompt early** (cosine 0.98 by step 100 in both A and B): the latent path
   carries almost no per-question variance for any loss to use. Strong evidence: geometry, flat CE, KL ≈ 0.
2. **The plan-reconstruction decoder is the full 4B backbone.** It lowers CE through its own plan-language prior
   and teacher forcing instead of through Z. Strong evidence: equal CE drops for all conditions, generic free
   reconstructions, and learned ≈ shuffled already at step 20 of the tiny overfit.
3. **Explicit-format forgetting** is a separate cost of training without any explicit-path term. In B, the
   plan-reconstruction branch shifts the model toward untagged plan-first output and repetition. Strong evidence
   for the format; the mechanism of B's repetition is untested.

## Limitations
- One seed, 300 steps (≈0.1 epoch), λp = 0.5 untuned.
- 50 generation problems: SE about ±6 points.
- Content tokens are rule-based, not annotated method spans.
- One deterministic Z(Q) per question can only test question-specific information, never strategy-specific planning.

## Recommended single next experiment (not run)
See `NEXT_STEPS.md`: freeze the backbone inside the plan-reconstruction branch, so that only Z can lower that CE.
