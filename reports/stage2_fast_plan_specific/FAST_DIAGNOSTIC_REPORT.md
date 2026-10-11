# Stage 2 fast plan-specific latent diagnostic (2026-10-11)

**Verdict: NO_CONDITIONAL_PLAN_SIGNAL** (under this evaluation). The plans are not template-dominated (not
LOW_PLAN_SPECIFICITY). Pilot B's latents do not improve prediction of the problem-specific plan tokens any more than
another question's latents, or the mean latent, do.

**Setup**
- Model: B `checkpoint-300` (L_exec + 0.5·L_planrec).
- Records: the first 128 val records, the same ids as the earlier diagnostics (hash verified).
- Held fixed: plan P, teacher-forced prefix, targets, mask and positions. Only Z changes.
- Implementation checks reused from the previous diagnostics and not rerun: causal mask, E/A leakage, and whether the
  swapped Z reaches the branch.
- No training. `eval()` + `inference_mode()`.

## P0: plan content audit (200 train, seed 42; 128 val)
| PlanRec target tokens | train (35,447 tokens) | val (23,086 tokens) |
|---|---|---|
| generic_template | 29.9% | 30.4% |
| method_specific | 2.2% | 2.4% |
| subgoal_specific (math-object proxy) | 27.7% | 27.0% |
| unknown | 40.2% | 40.2% |
| records with ≥1 method / subgoal / any specific span | 78% / 86% / 96% | 81% / 84% / 95% |
| records with specific share ≥10% | 74% | 72% |
| exact duplicate plans | 1% | 0% |
| answer string appears in plan (candidate) | 4.5% | 4.7% |
| ≥3 numeric equalities in plan (computation candidate) | 17% | 11% |
| plan 4-gram overlap with execution / plan lines restated verbatim | 0.75 / 69% | 0.77 / 67% |

- Specific spans cover about 29% of plan tokens and 95% of records, so the plans do carry problem-specific content.
- Label confidence is **MEDIUM-LOW**: the rule-based spans have good precision but low recall (see
  `span_annotation_policy.md` and the 24 examples in `p0_examples.md`). The true specific share is likely higher.
- Plans are heavily restated in the execution (67–69% of lines), a known property of this data.

## P1: span-level conditionality
Denominators: 121 records / 6,784 specific tokens (method 104 / 552, subgoal 108 / 6,232), generic 128 / 7,021,
full 128 / 23,086. The full CE reproduces Test 2 exactly (0.70696).

| CE | learned | shuffled s0 / s1 / s2 | mean | zero |
|---|---|---|---|---|
| full | 0.70696 | 0.70716 / … | 0.70687 | 0.70780 |
| specific (method ∪ subgoal) | 0.50611 | 0.50615 / … | 0.50574 | — |
| method | 1.06804 | 1.06781 / … | 1.06626 | — |
| subgoal | 0.45634 | 0.45640 / … | 0.45609 | — |
| generic (negative control) | 0.42094 | 0.42116 / … | 0.42070 | — |

Gains are **CE(cond) − CE(learned)**, so positive means the learned latents help. Paired bootstrap over questions,
2,000 resamples, seed 0.

| gain, 95% CI | token-weighted | per-question mean (median, share of questions > 0) |
|---|---|---|
| specific, shuffled (3-seed avg) | −0.00012 [−0.00054, +0.00026] | −0.00014 [−0.00153, +0.00131] (median −0.00008, 0.50) |
| specific, mean | −0.00037 [−0.00083, +0.00003] | −0.00080 [−0.00232, +0.00054] (median −0.00042, 0.42) |
| method, shuffled (3-seed avg) | −0.00005 [−0.00183, +0.00172] | −0.00007 [−0.00230, +0.00232] |
| method, mean | −0.00178 [−0.00387, +0.00022] | −0.00338 [−0.00706, +0.00002] |
| subgoal, shuffled (3-seed avg) | −0.00013 [−0.00057, +0.00026] | +0.00057 [−0.00093, +0.00225] |
| subgoal, mean | −0.00025 [−0.00073, +0.00021] | +0.00029 [−0.00138, +0.00217] |
| generic, shuffled (3-seed avg) | +0.00019 [−0.00013, +0.00051] | +0.00020 [−0.00012, +0.00051] |
| full, shuffled (3-seed avg) | +0.00020 [−0.00004, +0.00044] | +0.00015 [−0.00010, +0.00041] |

- **No positive conditional gain on specific tokens.** All learned-vs-shuffled and learned-vs-mean CIs include 0; the
  point estimates are slightly negative.
- Only the out-of-distribution zero latent is worse.
- **Sanity check** (`p1_specific_token_examples.md`, 15 records): the log-probabilities of specific tokens are almost
  identical under learned and shuffled latents (e.g. `inequality` −0.038 / −0.036, `probability` −1.738 / −1.738).
  Low CE on math tokens comes from copying within the teacher-forced plan, not from Z.
- **Validation** (`p1_eval_validation.json`): target ids match the offset tokenisation for all 128 records; token
  counts are equal across conditions; 12 records' specific tokens all decode to text in their plan.

## P2: SKIPPED_NO_RELIABLE_LABELS (`P2_SKIPPED.md`)

## Limitations
- Spans are rule-based (medium-low confidence, low recall). One model, one seed, 300 steps.
- A teacher-forced prefix can reveal later content, so even a positive gain would only show conditional dependence.
- This negative result, together with the earlier Test 2, supports a generic soft prompt for this checkpoint. It does
  not show that every training scheme would fail.

## Next step
One proposal in `NEXT_STEPS.md`. Nothing was started.
