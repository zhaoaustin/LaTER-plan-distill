# Stage 2 two-loss PlanRec pilot: report (2026-10-11)

## Conclusion: FAIL (task-sheet Gate, section 8: "PlanRec learned ≈ shuffled/mean")
Within 300 steps, `L_exec + 0.5·L_planrec` did not make the latents carry question-specific plan information.
- **PlanRec CE fell by 13%** (0.80 → 0.70), but equally for learned, shuffled, mean and zero latents.
- **Learned vs shuffled/mean on the 200-record held-out set:**
  - PlanRec gain +0.00015, 95% CI [−0.00008, +0.00041];
  - execution gain −0.00002, CI [−0.00010, +0.00005].
  - Every CI includes 0.
- **Z collapsed to an almost question-independent vector** in both runs: cross-question cosine 0.785 → 0.984 (B) and
  0.969 (A).
- **Plans reconstructed from Z alone are generic.** For 16 val records the learned latents produce only 2 distinct
  plans, both about "the total number of students".
- **The PlanRec branch became a plan language model that ignores Z**, which is the shortcut the task sheet warned about.
- **Extension:** per the sheet, no 500-step extension (it requires a positive sign) and no 1-epoch run.
- **Limitation:** each question has a single plan, so this pilot could only test question-conditioned information.
  Strategy-specific planning would need multi-plan data.

## Setup (one variable changed)
| | A: execution only | B: execution + PlanRec |
|---|---|---|
| loss | L_exec | L_exec + 0.5·L_planrec |
| init | Stage-1 step 800 (`a6b0e2f`) + seed-1234 projector | identical (hashes 268ae2db… / 8d73d82e…) |
| data order | DistributedSampler seed 1234 | identical per step and rank (verified) |
| steps / global batch / examples | 300 / 16 / 4,800 (≈0.10 epoch) | same |
| LR | 1e-5 backbone, 1e-4 projector, warmup 10, cosine horizon 500 | same |
| W&B (`stage2-plan-reconstruction`, group `planrec_two_loss_pilot`) | `bacf7c0020df` | `43fadd49fb35` |
| train time / step time / peak memory | 21.8 min / 4.35 s / 106 GiB | 23.8 min / 4.76 s / 115 GiB |

**PlanRec branch.** A separate causal forward through the same Qwen3 backbone with context
`[z1..z8] + "<EXPLORATION>\n" + teacher-forced plan tokens` and fresh positions. It never sees Q, E or A, and it does
not reuse the question's KV cache.
- Targets are the Stage-1 plan-segment tokens (they decode to plan + "\n").
- Z is the learned latent sequence inserted into the execution forward, with its gradient graph kept.
- Disabled and validated off: alignment, explicit CE, execution or teacher KL, halting, noise, RL. No teacher cache is
  loaded.

Code: branch `exp/stage2-planrec-two-loss`; training ran at commit `727c016`.
Data: 46,510 / 2,446 records, all of which reconstruct `stage1_solution` losslessly; hashes are in `dataset_manifest.json`.

## Gates passed before the pilot
- **Unit tests:** 10 new PlanRec tests plus the 30 existing three-loss tests, all passing. They cover:
  - masks: PlanRec targets = plan segment only; execution targets = E + A + closing;
  - leakage: Z is unchanged when the plan or execution is changed, and changes with Q;
  - causality, and the first/last-token shift checked against a manual CE;
  - gradients of L_planrec alone reaching z1, the projector and the backbone;
  - L_total exactly equal to the sum of the two terms, with A never computing the PlanRec branch;
  - override latents reproducing the builder (learned, zero and shuffled);
  - the monitor being read-only.
- **DDP and resume:**
  - the two ranks see disjoint records, 16 per step, and end with identical parameter hashes;
  - 2 steps + resume + 2 steps uses exactly the same batches as 4 straight steps;
  - the losses differ by the same amount as two identical straight runs (non-deterministic SDPA backward, about 0.002).
- **Tiny overfit** (16 train records, 40 steps):
  - PlanRec CE 0.744 → 0.012;
  - the first 32 content tokens: learned 0.064 vs shuffled 0.116;
  - free reconstruction from Z picks the record's own plan in 7/16 cases with learned latents, vs 1/16 with shuffled
    or mean latents.
  - Already at step 20, learned ≈ shuffled (0.040 vs 0.040): the backbone memorises the plans and teacher forcing
    supplies the identity. This was the early sign of the shortcut.

## Results: fixed first 200 val records, identical targets and masks for every condition
| | step 0 | A step 300 | B step 300 |
|---|---|---|---|
| exec CE learned / shuffled / mean / zero | 0.27573 / 0.27567 / 0.27601 / 0.27645 | 0.23616 / 0.23611 / 0.23613 / 0.23614 | 0.23629 / 0.23627 / 0.23635 / 0.23624 |
| **exec gain shuffled − learned** (95% CI) | −0.00006 [−0.00028, +0.00015] | −0.00004 [−0.00012, +0.00004] | −0.00002 [−0.00010, +0.00005] |
| exec gain mean − learned | +0.00027 [+0.00008, +0.00046] | −0.00002 [−0.00011, +0.00006] | +0.00005 [−0.00002, +0.00013] |
| PlanRec CE learned / shuffled / mean / zero | 0.8005 / 0.8002 / 0.7934 / 0.7830 | 0.9358 / 0.9361 / 0.9366 / 0.9342 | **0.7024 / 0.7026 / 0.7026 / 0.7031** |
| **PlanRec gain shuffled − learned** | −0.0004 [−0.0029, +0.0023] | +0.00022 [−0.00014, +0.00059] | +0.00015 [−0.00008, +0.00041] |
| PlanRec gain mean − learned | −0.0072 [−0.0091, −0.0051] | +0.00075 [+0.00042, +0.00110] | +0.00015 [−0.00007, +0.00037] |
| first-32 content-token CE, learned / shuffled | 1.5757 / 1.5752 | 1.5904 / 1.5913 | 1.3422 / 1.3423 |
| content32 gain shuffled − learned | −0.0005 [−0.0041, +0.0032] | +0.0009 [−0.0001, +0.0019] | +0.0002 [−0.0005, +0.0009] |
| cross-question cosine of Z (raw / mean-centred) | 0.785 / −0.005 | 0.969 / −0.004 | 0.984 / −0.004 |
| between-question std / norm of Z | 0.463 | 0.176 | 0.127 |

Paired bootstrap: 4,000 resamples over records.
- The table shows token-weighted values. Per-example means agree in sign and significance; both are in `metrics.json`.
- Content tokens are plan tokens outside the header line, step numbering and whitespace (83% of plan tokens).
- In A, PlanRec CE rises (0.80 → 0.94) because A never trains that branch and the backbone drifts. A's small,
  significant mean-vs-learned gap (+0.0008) is far below anything meaningful.

**Gradients.**
- At step 0 the weighted PlanRec gradient on the projector is 1.7–5.6× the execution gradient, depending on the
  batch. The sheet's balance check was recorded, and λp was not changed.
- During B, the projector gradient norm fell from 0.86 (mean over steps 1–50) to about 0.01 (steps 150–250), with a
  spike over steps 251–300 (mean 1.2).
- PlanRec's gradient on z1 stayed 3–28× the execution gradient, but absolutely small (0.005–0.10).
- Once Z was near-constant, the decoder reduced CE through its own weights rather than through Z.

**Free generation** (50 MATH-500 problems, 13-gram-clean, greedy, max 3,072 tokens; Q → Z → E,A, PlanRec branch unused):
| accuracy / completion | learned | mean | shuffled | zero | explicit Q→P→E |
|---|---|---|---|---|---|
| step 0 | 0.66 / 0.84 | 0.76 / 0.88 | 0.72 / 0.86 | 0.68 / 0.84 | 0.72 / 0.92 |
| A step 300 | 0.76 / 0.94 | 0.80 / 0.98 | 0.76 / 1.00 | 0.76 / 0.98 | **0.00 / 0.00** |
| B step 300 | 0.82 / 1.00 | 0.80 / 0.94 | 0.82 / 0.98 | 0.80 / 0.98 | **0.00 / 0.00** |

- **Latent-path accuracy rose with training** in both runs, but shuffled, mean and zero latents score the same. The
  gain comes from execution training, not from the latents. The standard error is about ±6 points.
- **Explicit mode is broken in both runs**, a cost of dropping L_explicit (the old three-loss setup kept it at 0.66–0.68).
  - A skips `<EXPLORATION>` and writes execution text directly.
  - B writes the plan without the tag, then `Answer: \boxed{…}`, then the steps.
  - The parser finds neither format, so A and B both get 0/50.
- **Latency, B learned:** 2.9 s per problem (batch 25), mean 487 tokens.

**Plan reconstruction from Z alone** (16 val records, greedy; diagnostic only): mean token F1 with the gold plan is
0.352 learned, 0.354 shuffled, 0.355 mean; recall of `$…$` expressions is 0 for all. Samples are in `samples.jsonl`.

## Failure cases
- **B, s1val-00000:** the question asks for the probability that three pieces of a stick form a triangle. The learned Z
  reconstructs "1. Identify the total number of students and the number of students who received each grade…", and
  so do the shuffled and mean latents.
- **Explicit format:** both runs fail as described above.

## Should this continue to 1 epoch? No.
The gate result is FAIL. More epochs would add training without the missing signal, the same Z collapse that A shows.
Directions worth testing instead, one at a time (not run):
1. Stop the decoder from memorising the plan prior: freeze the backbone, or use a separate, weaker decoder in the
   PlanRec branch, so that the gradient has to go through Z.
2. Plan-specific contrastive supervision, or data with several plans per question (e.g. Stage-1 plan A/B), so that a
   question-independent Z cannot fit.
3. Keep L_explicit, or another explicit-format term, if explicit mode is to be preserved.

## Cost
| | time |
|---|---|
| implementation + unit tests | ≈2 h |
| trainer gates (DDP/resume 3 runs + tiny overfit) | ≈15 min |
| A + B training, 2× B200 | 22 + 24 min (+ probes) |
| final evaluation, 3 models in parallel on 2 GPUs | ≈55 min |
