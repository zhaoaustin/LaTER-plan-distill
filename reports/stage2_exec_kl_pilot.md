# Stage 2 Execution-KL controlled pilot (2026-10-10)

**Result: no evidence that execution-KL makes the latents useful within 300 steps.** B moved slightly closer
to the plan-conditioned teacher (exec-KL −0.8%, first-16 KL −1.1% vs A), but learned, zero and shuffled latents
remain indistinguishable in both groups. All paired 95% CIs include 0. In 4 of 5 examples the greedy continuations
with learned and zero latents are character-identical; in the fifth they diverge after 217 characters with no
change of strategy. This is a short-budget pilot. It does not show that the
method fails with full training. Machine-readable results: `stage2_exec_kl_pilot.json`.

## Setup (one variable changed)
| | A: three-loss baseline | B: execution-KL pilot |
|---|---|---|
| loss | L_exec + 0.1·L_align + 0.3·L_explicit | same + **0.3·L_exec_kl** (fixed, not tuned) |
| start | checkpoint-500 of the paused 2,900-step run | same |
| initial hashes (model / projector) | fdcc1d61… / 2d655073… | identical |
| steps / global batch | 500 → 800 (+300) / 16 | same |
| batches | record ids per step identical to the precomputed plan | identical |
| LR | cosine with horizon 2,900, from step 500 | same |
| W&B (group `stage2_exec_kl_pilot`) | `a9374e685a68` | `41a0355087ba` |

Start state: weights, AdamW state (all 402 tensors, step 500), step counter and torch RNG come from checkpoint-500.
The Python RNG was not saved in that checkpoint, which affects both groups identically. Training uses no RNG.

**Execution-KL.** Forward KL(teacher ‖ student) at temperature 1.
- Teacher (frozen Stage 1, step 800) sees Q + P + boundary + E<t. Student sees Q + z1..8 + the same boundary + E<t.
- Positions are aligned by execution offset.
- Targets: the first 128 execution tokens of segment E that are not on plan-restatement lines (exact line match);
  answer tokens and closing tags are excluded.
- Every training record had KL targets, 127.1 on average (0 records without targets).
- The teacher is cached as top-128 probabilities plus an explicit tail bucket (no renormalisation). Top-128 mass is
  ≈ 1.0000 and the largest tail is 0.003. The cache covers only the 4,800 training records of steps 501–800; a
  separate val-probe cache is used for evaluation only.
- Execution CE (full E + A), explicit CE (P + E + A) and boundary alignment are unchanged.

## Gate (passed before training)
- 30/30 contract tests, 3 of them new for KL:
  - cached KL matches full-vocabulary KL;
  - the teacher's prediction at offset j depends only on E<j;
  - the KL mask is correct;
  - KL alone reaches the projector and the first latent;
  - L_total_B = A + 0.3·KL, and with KL off the losses are exactly the three-loss values;
  - a mismatched target raises an error.
- On checkpoint-500 (4 records): cache vs live full KL differs by 0.0003–0.002; gradient cosine 0.998–0.99995.
- Loss scales at start: L_exec 0.256, 0.1·L_align 0.0075, 0.3·L_explicit 0.059, 0.3·L_exec_kl 0.080.
- Projector gradients at start (weighted): exec 0.0010, align 0.00065, KL 0.0013.
  cos(KL, exec) = 0.76; cos(KL, align) = −0.37.
- Tiny run: 3 steps, save at 502, resume to 504; batches matched the plan. No NaN or OOM; peak 112 GiB.

## Results on the fixed 128 held-out val records (identical targets for every condition)
| | start (500) | A (800) | B (800) |
|---|---|---|---|
| CE learned / zero / shuffled | 0.23695 / 0.23687 / 0.23684 | 0.23541 / 0.23542 / 0.23547 | 0.23478 / 0.23487 / 0.23474 |
| non-repeat execution CE (learned) | 0.21241 | 0.21173 | 0.21079 |
| **gain zero − learned** (95% CI) | −0.00007 [−0.00021, +0.00006] | +0.00001 [−0.00012, +0.00014] | +0.00009 [−0.00006, +0.00025] |
| **gain shuffled − learned** (95% CI) | −0.00010 [−0.00021, +0.00001] | +0.00006 [−0.00004, +0.00017] | −0.00003 [−0.00014, +0.00007] |
| exec-KL teacher‖student (learned) | 0.12489 | 0.12473 | **0.12378** |
| first-16 KL (learned) | 0.27191 | 0.26899 | **0.26603** |
| exec-KL zero / shuffled | 0.12510 / 0.12496 | 0.12508 / 0.12472 | 0.12402 / 0.12382 |
| explicit execution CE (probe) | 0.15603 | 0.15636 | 0.15529 |
| alignment learned / zero | 0.0710 / 0.0827 | 0.0638 / 0.0749 | 0.0697 / 0.0817 |

Paired bootstrap: 4,000 resamples over records, token-weighted. Records with a positive gain: 0.49 at start,
0.56 for A and 0.53 for B.

**Projector gradient contributions (weighted, fixed 4-record batch):**
| step | A exec / align | B exec / align / KL | B cos(KL, exec) / cos(KL, align) |
|---|---|---|---|
| 600 | 0.00063 / 0.00046 | 0.00406 / 0.00047 / 0.00414 | 0.92 / −0.20 |
| 700 | 0.00372 / 0.00030 | 0.00077 / 0.00025 / 0.00063 | 0.55 / −0.09 |
| 800 | 0.00045 / 0.00032 | 0.00119 / 0.00051 / 0.00232 | 0.50 / −0.15 |

KL and execution CE push the projector in similar directions. KL and the boundary alignment are mildly opposed at
every probe point.

**Generation (50 MATH-500 problems, 13-gram-clean vs the E2C SFT source; sanity check only):**
| | learned | zero | shuffled | explicit |
|---|---|---|---|---|
| A step 800 | 0.74 | 0.78 | 0.74 | 0.66 |
| B step 800 | 0.76 | 0.70 | 0.74 | 0.68 |

One problem is 2 points and the standard error is about ±6 points, so none of these differences is meaningful.
Explicit accuracy on this subset is lower than the latent conditions in both groups. The probe's explicit CE did
not degrade (0.156 → 0.156 / 0.155).

**Examples (B step 800).** 5 records: the 2 largest per-record gains, the median, and the 2 most negative; full
text is in the JSON.
In 4 of 5 the 96-token greedy continuation with learned latents is **character-identical** to the one with zero
latents. In s1val-00049 they share the first 217 characters and then word one sentence differently, with no change
of strategy. All of them differ from the teacher with the explicit plan, e.g. in s1val-00049 the teacher assumes `a*b = a/b` while
both latent conditions start "Recognize the operation is left-associative…".

## Decision (per the task sheet)
- **KL fell slightly, but learned ≈ zero ≈ shuffled** in both groups. The KL term changed the output behaviour a
  little, and it is not carried by the latents: zero and shuffled latents improved by the same amount. So there is
  no evidence that a latent plan is being used, and this cannot be called a distillation success.
- B is not worse than A: explicit CE is preserved and execution CE is marginally lower.
- Conclusion: **short-budget pilot, no evidence.** This does not imply failure at full training. The data is still
  consistent with latent bypass. KL currently improves Q-conditioned execution, which zero and shuffled latents
  share, rather than plan transfer.

## Cost
| | time |
|---|---|
| teacher top-k cache (4,800 train records + 128 val) | 4 min |
| pilot A, 300 steps | 36 min |
| pilot B, 300 steps (+10 min failed attempt) | 39 min (+10) |
| shared | 5.15 s/step, peak 112–113 GiB/GPU on 2× B200 |

## Known issues / caveats
- The first B attempt crashed at step 600: the legacy val-loss evaluation did not skip exec-KL. It was fixed and
  B was rerun from the identical start. The failed run is kept (`runs/pilot_B_exec_kl_FAILED_step600_evalbug`,
  W&B `8ce1a78de359`).
- 300 steps = 4,800 examples ≈ 0.1 epoch. λ_kl = 0.3 is untuned.
- KL is applied only to the first ≤ 128 eligible execution tokens.
- The val probe has 128 records. Effects under about 1.5e-4 CE cannot be resolved.
- The Stage-1 teacher saw 320 benchmark-overlapping records; the 50-problem set is 13-gram clean but paraphrases
  are not detected.
