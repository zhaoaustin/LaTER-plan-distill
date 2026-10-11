# Next steps (proposals only; nothing here has been run or started)

## Recommended single next experiment: freeze the backbone inside the plan-reconstruction branch
**Change.** In the plan-reconstruction forward only, run the Qwen3 backbone with frozen weights, either as a frozen
copy or with no parameter gradients from L_planrec. Everything else stays identical to pilot B: λp = 0.5, the same
data order, steps, seed and projector initialisation, and the A control. L_planrec gradients would then reach only
Z, and through Z the latent recursion.

**Root cause targeted (#2 in the report).** The 4B decoder lowered plan-reconstruction CE through its own plan prior
and teacher forcing, without using Z (learned ≈ shuffled, KL 1e-4). With the decoder frozen, the only way to reduce
L_planrec is to put question-specific information into Z. It directly tests whether the latent channel can carry
plan information at pilot scale.

**Why it beats more steps or a larger λp.**
- Longer training: the CE drop is already condition-independent at step 100 and stays so to step 300.
- Larger λp: it scales the same shortcut gradient, which mostly flows into the backbone.

**Pass criterion (pre-registered):**
- plan-reconstruction gain vs shuffled and vs mean > 0, with the 95% CI excluding 0, on the same 128 records, also at
  the first 16/32 tokens;
- free reconstructions that differ by question;
- then check whether the execution gain follows.

## Other candidates (one at a time, after the above)
- **Restore an explicit-path term** (L_explicit or a format-only term) as a controlled ablation, if explicit mode
  must be kept. Test 1 shows format collapse in both A and B without it, and math loss in B.
- **Plan-specific / contrastive reconstruction**, or data with several plans per question, to move from
  question-specific to strategy-specific information. A deterministic Z(Q) cannot distinguish several valid plans of
  one question without an extra conditioning input.
- **Investigate the early Z collapse** (cosine 0.98 by step 100, also in A): projector LR, gradient clipping,
  projector normalisation. This is diagnostic only and needs its own controlled runs.

Not proposed: alignment, KL, Gaussian noise, RL, or a 2,900-step run. None addresses the observed root causes, and
none should be started without a decision.
