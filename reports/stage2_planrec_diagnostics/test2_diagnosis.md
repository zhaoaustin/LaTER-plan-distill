# Test 2: does plan reconstruction depend on the correct latent?

**Verdict for B: GENERIC_SOFT_PROMPT.** The implementation checks pass, so the result is interpretable.

## Setup
- **Records:** the first 128 records of `stage2_val` (the pilot probe set).
- **Held fixed:** the reference plan P, the plan-reconstruction target mask, the execution targets, latent length 8,
  positions and attention masks.
- **Varied:** only the latents.
- **Plan-reconstruction branch:** a fresh forward (no KV cache reuse) with context `[z] + "<EXPLORATION>\n" + P<t`.
- **Bootstrap:** paired over records, 2,000 resamples, seed 0, token-weighted (per-record means agree; see
  `metrics.json`).
- **Content tokens:** the rule fixed before the pilot (non-header, non-numbering, non-whitespace; coverage 83.6% of
  plan tokens). These are "non-template tokens", not annotated method or subgoal spans; method-span CE is
  **not_evaluated**.

## B step 300
| CE | learned | shuffled s0 / s1 / s2 | mean | zero |
|---|---|---|---|---|
| plan reconstruction, all | 0.70696 | 0.70716 / 0.70716 / 0.70717 | 0.70687 | 0.70780 |
| first 16 plan tokens | 0.54453 | 0.54482 / 0.54444 / 0.54487 | 0.54425 | 0.54705 |
| first 32 | 0.94545 | 0.94533 / 0.94531 / 0.94542 | 0.94529 | 0.94721 |
| content | 0.83507 | 0.83532 ×3 | 0.83501 | 0.83585 |
| first 32 content | 1.31163 | 1.31157 / 1.31174 / 1.31203 | 1.31189 | 1.31272 |
| execution (E + A) | 0.23105 | 0.23115 / 0.23113 / 0.23108 | 0.23108 | 0.23098 |
| KL(learned‖cond), first 16 plan positions | 0 | 0.00013 | 0.00010 | 0.00251 |

| gain = CE(cond) − CE(learned), 95% CI | shuffled (3-seed mean) | mean |
|---|---|---|
| plan reconstruction, all | +0.00020 [−0.00004, +0.00044] | −0.00010 [−0.00037, +0.00019] |
| first 16 | +0.00018 [−0.00037, +0.00071] | −0.00028 [−0.00083, +0.00023] |
| first 32 | −0.00010 [−0.00070, +0.00048] | −0.00016 [−0.00086, +0.00050] |
| content | +0.00025 [−0.00003, +0.00054] | −0.00006 [−0.00038, +0.00027] |
| first 32 content | +0.00015 [−0.00062, +0.00094] | +0.00025 [−0.00053, +0.00107] |
| execution | +0.00007 [−0.00001, +0.00015] | +0.00003 [−0.00006, +0.00011] |

- **Every learned-vs-shuffled and learned-vs-mean CI includes 0.** The largest point estimates are ≤ 0.03% of CE.
  This holds at the plan start (first 16/32), so teacher forcing is not hiding a start-of-plan effect.
- **The decoder output barely moves with Z:** KL 1.3e-4 nats. Without any plan prefix, the first plan token's
  distribution differs by 7e-8 nats between learned and shuffled latents, with top-1 `I` in all 20 records.
- **Zero latents are slightly worse** (+0.0008 overall) only because they are out of distribution.
- **Latent geometry:**
  - cross-question cosine per slot 0.975–0.996, flattened 0.984, mean-centred −0.007;
  - RMS = token-embedding RMS;
  - between-question variance is 0.5–2.5% of latent energy.
  - Z is nearly a constant soft prompt.

## Free reconstruction (B, 20 records, from `"<EXPLORATION>\n"` only; raw text in `free_planrec_samples.jsonl`)
| | learned | shuffled s0 | mean | zero |
|---|---|---|---|---|
| distinct outputs / 20 | 2 | 2 | 1 | 1 |
| picks own plan among the 20 (token-F1 argmax) | 1 | 1 | 1 | 1 |
| mean token F1 with own gold plan | 0.350 | 0.349 | 0.353 | 0.347 |
| ends with `</` | 1.00 | 1.00 | 1.00 | 1.00 |
| repeated-line fraction across outputs | 0.93 | 0.93 | 0.95 | 0.95 |

All 20 learned reconstructions start "I need to make a step-by-step plan.\n1. Identify the total number of students
and the number of students who received each grade…", which is unrelated to every question. Strategy accuracy was
not scored (undetermined); the outputs are visibly generic.

## A step 300 (auxiliary; A never trained the plan-reconstruction branch)
- Plan-reconstruction CE is about 0.944 for all latent conditions; every learned-vs-shuffled and learned-vs-mean CI
  includes 0.
- Execution: shuffled/mean − learned is −0.00009 / −0.00014. CI excludes 0 but is negative: the learned latents are
  marginally worse.
- Cosine 0.969.
- A is also question-independent.

## Implementation checks on these checkpoints (`causal_mask_tests.json`): all passed
1. **Future plan token changed:** earlier logits differ by 0.0, later logits by 15.6.
2. **E/A edited:** the hash of Z is unchanged.
3. **Swapped Z reaches the branches:** the latent tensor inserted into the execution forward has the hash of the
   swapped Z, not the learned Z. Plan-reconstruction logits change by up to 1.3, and execution NLL by up to 0.02.
4. **No question tokens** in the plan-reconstruction context (32 records).

## Interpretation (supported vs not)
- **Supported:** after 300 steps, B's decoder reconstructs plans with no causal dependence on the correct question's
  latent. The 13% plan-reconstruction CE drop during the pilot came from the decoder's language-model prior on plan
  text, plus teacher-forced prefixes after the first tokens.
- **Supported by the tiny overfit, not by this test:** the branch can learn Z dependence when the backbone memorises a
  handful of plans (7/16 identification).
- **Not established:** why Z collapsed (projector LR, clipping with pre-clip gradient norms of about 2–50, or the generic
  language-model solution simply being easier). Testing that needs training, which was out of scope.
