# Next step (proposal only; not started)

**One experiment: a PlanRec loss restricted to specific spans, with the backbone frozen inside the PlanRec branch.**

- **Target.** Compute L_planrec only on the P0 method and subgoal tokens (the frozen rules; about 29% of plan
  tokens, 95% of records). Run the PlanRec forward with a frozen backbone, so that only Z (and the latent recursion)
  receives that gradient.
- **Controls.** Keep everything else as in pilot B: λp, data order, steps, seed, A control.
- **Why.**
  - P1 shows the problem is not that specific content is missing from the targets (it is present and covered).
    Rather, nothing in training forces Z to carry it.
  - The full-plan CE is dominated by generic and copyable tokens that the 4B decoder predicts from its own prior.
  - Restricting the targets removes the template share, and freezing the decoder removes the language-model
    shortcut. This is more informative than longer training or a larger λp, both of which scale the same shortcut.
- **Pre-registered pass criterion:** specific-token gain vs shuffled and vs mean > 0 with the 95% CI excluding 0 on
  the same 128 records. Then check whether the execution gain follows.
