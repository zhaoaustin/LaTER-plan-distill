# Stage 2 forward-only diagnostics at step 500 (2026-10-10)

Checkpoint: runs/stage2_L8_rms_2900/checkpoint-500 (training paused after a verified resumable save; see
resume_manifest_step500.json). Fixed set: first 128 records of stage2_val; teacher forcing on gold E; "nonrep" =
execution tokens not on plan-restatement lines. Scripts: later/scripts/train/stage2_diagnostics.py.

## Test A — Stage-2 latent sensitivity (same L, positions, boundary)
| | CE | CE nonrep |
|---|---|---|
| learned | 0.2369 | 0.2124 |
| zero | 0.2369 | 0.2124 |
| shuffled | 0.2368 | 0.2124 |
| Stage-2 student, gold plan (explicit) | 0.1560 | 0.1732 |

Next-token KL(learned‖zero) = 0.00027 (first 16 exec tokens 0.0018), top-1 agreement 99.7%;
KL(learned‖shuffled) = 0.00017, top-1 agreement 99.75%. KL(explicit‖learned) = 0.082.
The latents have essentially no effect on the execution distribution.

## Test B — Stage-1 teacher plan A/B intervention (128 records)
| plan given to teacher | CE(E_A) | nonrep | KL(A‖·) | KL first 16 tok |
|---|---|---|---|---|
| A gold | 0.142 | 0.158 | – | – |
| B same steps reversed | 0.162 | 0.169 | 0.022 | 0.349 |
| M other record's plan | 0.287 | 0.230 | 0.147 | 1.537 |
| N empty | 0.274 | 0.225 | 0.133 | 0.959 |

Strategy switching (greedy, 64 records): with B the first executed step follows B in 73% (A order 9%), full step
order matches B in 58% (A: 88% match A). The teacher follows the given plan.
Teacher boundary distance to A (normalized L1, mean of 36 layers): B 0.060, M 0.088, N 0.267.

## Test C — boundary-to-execution transfer (activation patching of the teacher boundary hidden)
| model / patch | CE | gap to gold-plan CE closed (all / nonrep) |
|---|---|---|
| Stage-2 learned, patch all 36 layers | 0.2366 | 0.4% / 1.0% |
| Stage-2, patch layers 1–12 / 13–24 / 25–36 | 0.2367 / 0.2366 / 0.2368 | ≤ 0.4% |
| Stage-2, single layer 1/9/18/27/36 | 0.2368–0.2369 | ≤ 0.2% |
| Stage-1 empty plan, patch all 36 layers | 0.2724 (empty 0.2726, gold 0.1432) | 0.1% / 2.2% |

## Per-layer boundary alignment to the teacher (normalized L1)
| condition | mean | L1 | L9 | L18 | L27 | L36 |
|---|---|---|---|---|---|---|
| learned | 0.0711 | 0.042 | 0.059 | 0.062 | 0.082 | 0.127 |
| zero | 0.0828 | 0.046 | 0.078 | 0.068 | 0.090 | 0.143 |
| shuffled | 0.0711 | 0.042 | 0.059 | 0.063 | 0.083 | 0.127 |
| Stage-2 explicit path | 0.1251 | 0.081 | 0.124 | 0.110 | 0.140 | 0.187 |

## Reading
1. The teacher uses the plan strongly, mostly in the first execution tokens, and switches strategy when the plan
   changes. The plan signal is real.
2. That signal does not live in the single boundary position: plans that change execution CE by 2x move the
   teacher boundary hidden by only 0.06–0.09, and writing the teacher boundary hidden into the student (or into the
   Stage-1 model without a plan) closes ≤ 2% of the gap. Execution reads the plan through attention to plan tokens.
3. Hence L_alignment as specified (one boundary position) can be satisfied from the question alone (student 0.071
   ≈ teacher A-vs-B distance) and, even if satisfied perfectly, cannot transfer plan information to execution.
   The latents are not sample-specific: shuffled latents give the same alignment as the model's own latents.
4. Continuing the current 2,900-step run is not expected to change this; the objective needs to change.
