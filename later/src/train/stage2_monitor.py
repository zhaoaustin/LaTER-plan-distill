"""Monitoring for OUR Stage 2 (read-only: never updates parameters, never writes .grad, no optimizer step).

  probe_ce          fixed val records, identical execution targets/masks for every condition:
                    learned / zero / shuffled latents (same L, same positions, same boundary), no_latent (L = 0,
                    auxiliary), explicit (gold plan; execution-side targets only). Also the alignment loss for
                    learned / zero, and execution CE without plan-restatement tokens ("nonrep").
  projector_grad_split  ||dL_exec/dθ_proj|| and ||dL_align/dθ_proj|| on one fixed batch via autograd.grad.
  generation_eval   greedy decoding on a fixed contamination-checked math set: learned / zero / shuffled / explicit.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.distributed as dist

from later.src.train.losses import stage2_normalized_l1_alignment
from later.src.train.stage2_plan_dataset import Stage2Collator, build_views, render_prompt, stage2_tokens


def _dist():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


class Stage2Monitor:
    def __init__(self, model, tokenizer, val_frame, val_cache, layer_ids: List[int], latent_steps: int,
                 probe_n: int = 128, probe_batch_size: int = 8, grad_batch_size: int = 4, max_length: int = 4096):
        self.model, self.tok, self.cache, self.layer_ids = model, tokenizer, val_cache, list(layer_ids)
        self.L = int(latent_steps)
        T = stage2_tokens(tokenizer)
        collate = Stage2Collator(tokenizer.pad_token_id)
        rows = [val_frame.iloc[i].to_dict() for i in range(min(probe_n, len(val_frame)))]  # fixed: first N records
        self.probe_ids = [r["record_id"] for r in rows]
        self.batches, self.batches0 = [], []
        for s in range(0, len(rows), probe_batch_size):
            chunk = rows[s: s + probe_batch_size]
            if len(chunk) < 2:
                break
            self.batches.append(collate([build_views(tokenizer, r, self.L, T, max_length) for r in chunk]))
            self.batches0.append(collate([build_views(tokenizer, r, 0, T, max_length) for r in chunk]))
        self.grad_batch = collate([build_views(tokenizer, r, self.L, T, max_length) for r in rows[:grad_batch_size]])

    # --------------------------------------------------------------------------------------------
    def _forward(self, b, view, mode="learned", layers=(1,), exec_only=False):
        dev = next(self.model.parameters()).device
        labels = b[f"{view}_labels"].clone()
        if exec_only:
            keep = b[f"{view}_seg_E"] | b[f"{view}_seg_A"] | b[f"{view}_seg_C"]
            labels[~keep] = -100
        kw = dict(input_ids=b[f"{view}_input_ids"].to(dev), attention_mask=b[f"{view}_attention_mask"].to(dev),
                  position_ids=b[f"{view}_position_ids"].to(dev), labels=labels.to(dev),
                  boundary_positions=b[f"{view}_boundary"].to(dev), layer_ids=list(layers))
        if view == "student":
            kw.update(latent_positions=b["latent_positions"].to(dev),
                      latent_start_positions=b["latent_start_positions"].to(dev), latent_mode=mode)
        out = self.model.forward_plan_distill_stage2(**kw)
        rep = b[f"{view}_seg_R"][:, 1:].to(dev)[out["target_mask"]]
        return out, rep

    @torch.no_grad()
    def probe_ce(self) -> Dict[str, float]:
        rank, world = _dist()
        dev = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()
        names = ["learned", "zero", "shuffled", "no_latent", "explicit"]
        # [nll, tok, nll_nonrep, tok_nonrep] per condition + [align_sum, n] for learned / zero
        acc = torch.zeros(len(names), 4, device=dev, dtype=torch.float64)
        al = torch.zeros(2, 2, device=dev, dtype=torch.float64)
        try:
            for bi in range(rank, len(self.batches), world):
                b, b0 = self.batches[bi], self.batches0[bi]
                th = self.cache.get(b["record_ids"], self.layer_ids, dev)
                for ci, name in enumerate(names):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        if name == "no_latent":
                            out, rep = self._forward(b0, "student")
                        elif name == "explicit":
                            out, rep = self._forward(b, "explicit", exec_only=True)
                        else:
                            out, rep = self._forward(b, "student", name, layers=self.layer_ids if name != "shuffled" else (1,))
                    nll = out["token_nll"].double()
                    acc[ci] += torch.stack([nll.sum(), torch.tensor(float(nll.numel()), device=dev, dtype=torch.float64),
                                            nll[~rep].sum(), (~rep).sum().double()])
                    if name in ("learned", "zero"):
                        a, _ = stage2_normalized_l1_alignment(out["boundary_hidden"], th, self.layer_ids)
                        n = b["student_input_ids"].size(0)
                        al[0 if name == "learned" else 1] += torch.tensor([float(a) * n, n], device=dev, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(acc)
                dist.all_reduce(al)
        finally:
            self.model.train(was_training)
        res = {}
        for ci, name in enumerate(names):
            res[f"val/ce_{name}"] = float(acc[ci, 0] / acc[ci, 1])
            res[f"val/ce_{name}_nonrep"] = float(acc[ci, 2] / acc[ci, 3])
        res["val/latent_gain_ce"] = res["val/ce_zero"] - res["val/ce_learned"]
        res["val/latent_gain_ce_nonrep"] = res["val/ce_zero_nonrep"] - res["val/ce_learned_nonrep"]
        res["val/latent_gain_ce_vs_shuffled"] = res["val/ce_shuffled"] - res["val/ce_learned"]
        res["val/alignment_learned"] = float(al[0, 0] / al[0, 1])
        res["val/alignment_zero"] = float(al[1, 0] / al[1, 1])
        res["val/probe_exec_tokens"] = float(acc[0, 1])
        res["val/probe_nonrep_fraction"] = float(acc[0, 3] / acc[0, 1])
        return res

    def projector_grad_split(self, lambda_a: float) -> Dict[str, float]:
        """Rank 0 only. autograd.grad -> no .grad written, no optimizer step, no DDP reducer involvement."""
        dev = next(self.model.parameters()).device
        proj = [p for p in self.model.latent_projector.parameters()]
        was_training = self.model.training
        self.model.eval()
        try:
            b = self.grad_batch
            th = self.cache.get(b["record_ids"], self.layer_ids, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out, _ = self._forward(b, "student", "learned", layers=self.layer_ids)
                l_exec = out["token_nll"].float().mean()
                l_align, _ = stage2_normalized_l1_alignment(out["boundary_hidden"], th, self.layer_ids)
            g_exec = torch.autograd.grad(l_exec, proj, retain_graph=True, allow_unused=True)
            g_align = torch.autograd.grad(l_align, proj, allow_unused=True)
        finally:
            self.model.train(was_training)
        nrm = lambda gs: float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)))
        ge, ga = nrm(g_exec), nrm(g_align)
        return {"val/grad_projector_from_execution": ge, "val/grad_projector_from_alignment": ga,
                "val/grad_projector_from_alignment_weighted": lambda_a * ga,
                "val/grad_projector_align_over_exec": (lambda_a * ga) / ge if ge > 0 else float("nan")}

    @torch.no_grad()
    def generation_eval(self, problems: List[Dict[str, str]], conditions: List[str], max_new_tokens: int,
                        batch_size: int = 25) -> Dict[str, float]:
        from later.src.eval.eval_stage2 import analyse, verify  # same parser/verifier as the offline eval
        rank, world = _dist()
        T = stage2_tokens(self.tok)
        eos = [self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.convert_tokens_to_ids("<|endoftext|>")]
        mine = [p for i, p in enumerate(problems) if i % world == rank]
        was_training = self.model.training
        self.model.eval()
        rows = []
        try:
            for cond in conditions:
                explicit = cond == "explicit"
                for s in range(0, len(mine), batch_size):
                    chunk = mine[s: s + batch_size]
                    prompts = [self.tok.encode(render_prompt(self.tok, p["question"]), add_special_tokens=False)
                               + ([] if explicit else T.plan_open_ids) for p in chunk]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        g = self.model.generate_plan_latent(
                            prompts, 0 if explicit else self.L, [] if explicit else T.suffix_ids, max_new_tokens,
                            eos, self.tok.pad_token_id, latent_mode="learned" if explicit else cond)
                    for p, ids, stopped in zip(chunk, g["token_ids"], g["stopped"]):
                        a = analyse(self.tok.decode(ids, skip_special_tokens=False), explicit, stopped)
                        rows.append({"cond": cond, "correct": verify(a["pred"], p["answer"]),
                                     "complete": a["complete"], "len": len(ids)})
        finally:
            self.model.train(was_training)
        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, rows)
            rows = [r for part in gathered for r in part]
        res = {}
        for cond in conditions:
            rs = [r for r in rows if r["cond"] == cond]
            res[f"gen/acc_{cond}"] = sum(r["correct"] for r in rs) / len(rs)
            res[f"gen/complete_{cond}"] = sum(r["complete"] for r in rs) / len(rs)
            res[f"gen/mean_len_{cond}"] = sum(r["len"] for r in rs) / len(rs)
        res["gen/n_problems"] = len(problems)
        return res
