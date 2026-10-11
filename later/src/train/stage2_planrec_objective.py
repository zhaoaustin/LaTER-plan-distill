"""Stage 2 two-loss Plan Reconstruction objective (objective_mode=planrec_two_loss):

    L_total = execution_ce_weight * L_execution + lambda_p * L_planrec

  L_execution : CE over all E + A (+ closing tag/EOS) tokens of the latent path  Q -> z_1..z_L -> E, A
                (same view, labels and forward as the three-loss Stage 2)
  L_planrec   : CE over the Stage-1 plan tokens P, decoded by the SAME student backbone in a fresh causal forward
                whose context is only [z_1..z_L] + '<EXPLORATION>\\n' (+ teacher-forced earlier plan tokens).
                Z is the learned latent sequence of the execution forward, graph kept (no detach).
No alignment, no explicit-path CE, no teacher hidden/KL cache, no halting, no noise, no RL. Inference never runs
the PlanRec branch.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from later.src.train.losses import stage2_masked_mean_ce

DISABLED_FLOAT = ("lambda_a", "lambda_r", "kl_loss_weight", "lambda_kl", "noise_std", "halt_dense_loss_stage1_weight",
                  "halt_dense_loss_stage2_final_weight")
DISABLED_BOOL = ("teacher_kl_enabled", "execution_kl_enabled", "halting_loss_enabled", "rl_enabled")


def check_disabled(cfg: Dict[str, Any]) -> None:
    for k in DISABLED_FLOAT:
        if float(cfg.get(k, 0.0)) != 0.0:
            raise ValueError(f"{k} must be 0 for planrec_two_loss")
    for k in DISABLED_BOOL:
        if bool(cfg.get(k, False)):
            raise ValueError(f"{k} must be false for planrec_two_loss")


class PlanRecModule(nn.Module):
    """Single forward entry for DDP: the execution forward and the PlanRec forward run inside one call."""

    def __init__(self, model: nn.Module, cfg: Dict[str, Any]):
        super().__init__()
        check_disabled(cfg)
        self.model = model
        self.exec_w = float(cfg.get("execution_ce_weight", 1.0))
        self.lambda_p = float(cfg["lambda_p"])
        if self.exec_w <= 0:
            raise ValueError("execution_ce_weight must be > 0")

    def forward(self, batch: Dict[str, Any], debug_latent_grad: bool = False) -> Dict[str, Any]:
        dev = next(self.model.parameters()).device
        g = lambda k: batch[k].to(dev)
        lat = self.model.forward_plan_distill_stage2(
            input_ids=g("student_input_ids"), attention_mask=g("student_attention_mask"),
            position_ids=g("student_position_ids"), labels=g("student_labels"),
            boundary_positions=g("student_boundary"), layer_ids=[self.model.config.num_hidden_layers],
            latent_positions=g("latent_positions"), latent_start_positions=g("latent_start_positions"),
            debug_latent_grad=debug_latent_grad)
        l_exec, n_exec = stage2_masked_mean_ce(lat["token_nll"])
        z = lat["latent_embeddings"]                       # [B, L, d], same tensor that entered the exec forward
        out: Dict[str, Any] = {"latent": lat, "n_exec_tokens": n_exec, "z": z}
        if self.lambda_p > 0:
            rec = self.model.forward_plan_reconstruction(z, g("planrec_input_ids"), g("planrec_attention_mask"),
                                                         g("planrec_labels"))
            l_rec, n_rec = stage2_masked_mean_ce(rec["token_nll"])
            out.update(planrec=rec, n_planrec_tokens=n_rec)
        else:
            l_rec = torch.zeros((), device=dev)            # control run A: the branch is not computed at all
        total = self.exec_w * l_exec + self.lambda_p * l_rec
        out.update(loss=total, l_exec=l_exec, l_planrec=l_rec)
        return out
