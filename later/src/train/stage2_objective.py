"""OUR Stage 2 objective (objective_mode=plan_distill_stage2):

    L_total = execution_ce_weight * L_execution + lambda_a * L_alignment + lambda_r * L_explicit

  L_execution : CE over all E + A (+ closing tag/EOS) tokens of the student latent path  Q -> z_1..z_L -> E, A
  L_alignment : normalized multi-layer L1 between student boundary hidden (latent path) and frozen teacher
                boundary hidden (Q + P + boundary, cached), teacher std over d, eps
  L_explicit  : CE over all P + E + A tokens of the SAME trainable student on the explicit path Q -> P, E, A
No teacher KL, no halting loss, no noise, no RL.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from later.src.train.losses import (stage2_masked_mean_ce, stage2_normalized_l1_alignment, stage2_total_loss)


class Stage2Module(nn.Module):
    """Single forward entry for DDP: both student paths run inside one forward call."""

    def __init__(self, model: nn.Module, cfg: Dict[str, Any], layer_ids: List[int]):
        super().__init__()
        self.model = model
        self.layer_ids = list(layer_ids)
        self.lambda_a = float(cfg["lambda_a"])
        self.lambda_r = float(cfg["lambda_r"])
        self.exec_w = float(cfg.get("execution_ce_weight", 1.0))
        self.eps = float(cfg.get("alignment_epsilon", 1e-6))
        for k in ("kl_loss_weight", "halt_dense_loss_stage1_weight", "halt_dense_loss_stage2_final_weight",
                  "noise_std"):
            if float(cfg.get(k, 0.0)) != 0.0:
                raise ValueError(f"{k} must be 0 for plan_distill_stage2")
        for k in ("teacher_kl_enabled", "halting_loss_enabled", "rl_enabled"):
            if bool(cfg.get(k, False)):
                raise ValueError(f"{k} must be false for plan_distill_stage2")

    def forward(self, batch: Dict[str, Any], teacher_hidden: Dict[int, torch.Tensor] | None,
                debug_latent_grad: bool = False) -> Dict[str, Any]:
        dev = next(self.model.parameters()).device
        g = lambda k: batch[k].to(dev)
        zero = torch.zeros((), device=dev)
        out: Dict[str, Any] = {}
        run_latent = self.exec_w > 0 or self.lambda_a > 0
        if run_latent:
            lat = self.model.forward_plan_distill_stage2(
                input_ids=g("student_input_ids"), attention_mask=g("student_attention_mask"),
                position_ids=g("student_position_ids"), labels=g("student_labels"),
                boundary_positions=g("student_boundary"), layer_ids=self.layer_ids,
                latent_positions=g("latent_positions"), latent_start_positions=g("latent_start_positions"),
                debug_latent_grad=debug_latent_grad)
            l_exec, n_exec = stage2_masked_mean_ce(lat["token_nll"])
            out.update(latent=lat, n_exec_tokens=n_exec)
            if self.lambda_a > 0:
                if teacher_hidden is None:
                    raise ValueError("lambda_a > 0 but no teacher boundary hidden provided")
                l_align, align_info = stage2_normalized_l1_alignment(lat["boundary_hidden"], teacher_hidden,
                                                                     self.layer_ids, eps=self.eps)
                out["align_info"] = align_info
            else:
                l_align = zero
        else:
            l_exec, l_align = zero, zero
        if self.lambda_r > 0:
            exp = self.model.forward_plan_distill_stage2(
                input_ids=g("explicit_input_ids"), attention_mask=g("explicit_attention_mask"),
                position_ids=g("explicit_position_ids"), labels=g("explicit_labels"),
                boundary_positions=g("explicit_boundary"), layer_ids=self.layer_ids[-1:])
            l_explicit, n_x = stage2_masked_mean_ce(exp["token_nll"])
            out.update(explicit=exp, n_explicit_tokens=n_x)
        else:
            l_explicit = zero
        total = stage2_total_loss(l_exec, l_align, l_explicit, self.lambda_a, self.lambda_r, self.exec_w)
        out.update(loss=total, l_exec=l_exec, l_align=l_align, l_explicit=l_explicit)
        return out
