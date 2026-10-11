"""Monitoring for the two-loss PlanRec Stage 2 (read-only: no .grad writes, no optimizer step).

probe(): fixed val records (first N, identical tokenization / targets / masks for every condition).
  Z_learned is computed with the training builder for every probe record. Conditions, all injected through
  latent_override at the SAME L positions (the builder is not rerun, no learned KV cache is reused):
    learned  : each record's own Z
    zero     : all-zero vectors (auxiliary, may be out of distribution)
    mean     : per-slot mean of Z_learned over the N probe records (same vector for every record)
    shuffled : Z of another probe record (fixed derangement, seed 0, over the whole probe set)
  For every condition: execution CE (latent path, E + A + closing) and PlanRec CE (all plan tokens, content
  tokens, first 32 content tokens, math tokens). Token-weighted and per-example means; per-record values are
  returned for bootstrap CIs. Also cross-question cosine of Z (raw and mean-centred).
grad_split(): rank 0, one fixed batch, autograd.grad of L_exec and lambda_p * L_planrec w.r.t. projector params,
  all latents Z and z_1.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

import torch
import torch.distributed as dist

from later.src.train.stage2_plan_dataset import Stage2Collator, build_views, stage2_tokens

CONDITIONS = ("learned", "zero", "mean", "shuffled")


def _dist():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def fixed_derangement(n: int, seed: int = 0) -> List[int]:
    rng = random.Random(seed)
    while True:
        p = list(range(n))
        rng.shuffle(p)
        if all(p[i] != i for i in range(n)):
            return p


@torch.no_grad()
def build_latents(model, b) -> torch.Tensor:
    """Z_learned [B, L, d] with exactly the training builder (no grad)."""
    dev = next(model.parameters()).device
    lp = b["latent_positions"].to(dev)
    ls = b["latent_start_positions"].to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        over, mask, _ = model._build_latent_recursive_embeds(
            input_ids=b["student_input_ids"].to(dev), attention_mask=b["student_attention_mask"].to(dev),
            position_ids=b["student_position_ids"].to(dev), cot_mask=None, latent_positions=lp, latent_slot_mask=None,
            latent_start_positions=ls, global_prompt_len=int(ls.max()) + 1, use_no_grad=True)
    rows = torch.arange(lp.size(0), device=dev).unsqueeze(1)
    return over[rows, lp].float()


def cross_question_cosine(Z: torch.Tensor) -> Dict[str, float]:
    """Z [N, L, d]. Mean pairwise cosine (i != j) of flattened latents; per-slot mean; mean-centred variant."""
    N = Z.size(0)
    off = ~torch.eye(N, dtype=torch.bool, device=Z.device)

    def mpc(X):
        X = torch.nn.functional.normalize(X, dim=-1)
        return float((X @ X.T)[off].mean())
    flat = Z.reshape(N, -1)
    return {"latent_cross_question_cosine": mpc(flat),
            "latent_cross_question_cosine_per_slot": sum(mpc(Z[:, k]) for k in range(Z.size(1))) / Z.size(1),
            "latent_cross_question_cosine_centered": mpc(flat - flat.mean(0, keepdim=True)),
            "latent_between_question_std_over_norm": float(flat.std(0).norm() / flat.norm(dim=-1).mean())}


class PlanRecMonitor:
    def __init__(self, model, tokenizer, val_frame, latent_steps: int, probe_n: int = 128, probe_batch_size: int = 8,
                 grad_batch_size: int = 4, max_length: int = 4096, lambda_p: float = 0.5, first_k_content: int = 32):
        self.model, self.tok, self.L, self.lambda_p, self.K = model, tokenizer, int(latent_steps), float(lambda_p), first_k_content
        T = stage2_tokens(tokenizer)
        collate = Stage2Collator(tokenizer.pad_token_id)
        self.rows = [val_frame.iloc[i].to_dict() for i in range(min(probe_n, len(val_frame)))]
        self.probe_ids = [r["record_id"] for r in self.rows]
        self.batches = [collate([build_views(tokenizer, r, self.L, T, max_length) for r in self.rows[s: s + probe_batch_size]])
                        for s in range(0, len(self.rows), probe_batch_size)]
        self.grad_batch = collate([build_views(tokenizer, r, self.L, T, max_length) for r in self.rows[:grad_batch_size]])
        self.perm = fixed_derangement(len(self.rows))
        self.last_Z = None

    # ------------------------------------------------------------------------------------------------------------
    def _all_latents(self) -> torch.Tensor:
        return torch.cat([build_latents(self.model, b) for b in self.batches], dim=0)   # [N, L, d] fp32

    def _cond_latents(self, Z: torch.Tensor, cond: str, idx: List[int]) -> torch.Tensor:
        if cond == "learned":
            return Z[idx]
        if cond == "zero":
            return torch.zeros_like(Z[idx])
        if cond == "mean":
            return Z.mean(0, keepdim=True).expand(len(idx), -1, -1)
        if cond == "shuffled":
            return Z[[self.perm[i] for i in idx]]
        raise ValueError(cond)

    def _record_terms(self, b, z, dev) -> List[Dict[str, float]]:
        lat = self.model.forward_plan_distill_stage2(
            input_ids=b["student_input_ids"].to(dev), attention_mask=b["student_attention_mask"].to(dev),
            position_ids=b["student_position_ids"].to(dev), labels=b["student_labels"].to(dev),
            boundary_positions=b["student_boundary"].to(dev), layer_ids=[self.model.config.num_hidden_layers],
            latent_positions=b["latent_positions"].to(dev), latent_start_positions=b["latent_start_positions"].to(dev),
            latent_override=z)
        rec = self.model.forward_plan_reconstruction(z, b["planrec_input_ids"].to(dev), b["planrec_attention_mask"].to(dev),
                                                     b["planrec_labels"].to(dev))
        out = []
        e_cnt = lat["target_mask"].sum(1).tolist()
        r_cnt = rec["target_mask"].sum(1).tolist()
        rep = b["student_seg_R"][:, 1:].to(dev)[lat["target_mask"]]
        content = b["planrec_content"][:, 1:].to(dev)[rec["target_mask"]]
        mathm = b["planrec_math"][:, 1:].to(dev)[rec["target_mask"]]
        es, rs = 0, 0
        for i in range(len(e_cnt)):
            en = lat["token_nll"][es: es + int(e_cnt[i])].double()
            rp = rep[es: es + int(e_cnt[i])]
            rn = rec["token_nll"][rs: rs + int(r_cnt[i])].double()
            cm = content[rs: rs + int(r_cnt[i])]
            mm = mathm[rs: rs + int(r_cnt[i])]
            cn = rn[cm]
            out.append({"exec_sum": float(en.sum()), "exec_n": int(en.numel()),
                        "exec_nonrep_sum": float(en[~rp].sum()), "exec_nonrep_n": int((~rp).sum()),
                        "rec_sum": float(rn.sum()), "rec_n": int(rn.numel()),
                        "rec_content_sum": float(cn.sum()), "rec_content_n": int(cn.numel()),
                        "rec_content32_sum": float(cn[: self.K].sum()), "rec_content32_n": int(min(self.K, cn.numel())),
                        "rec_math_sum": float(rn[mm].sum()), "rec_math_n": int(mm.sum())})
            es += int(e_cnt[i]); rs += int(r_cnt[i])
        return out

    @torch.no_grad()
    def probe(self, return_records: bool = False) -> Dict[str, Any]:
        rank, world = _dist()
        dev = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()
        recs: Dict[str, Dict[str, Dict[str, float]]] = {}
        try:
            Z = self._all_latents()                       # every rank: all records (needed for mean / shuffled)
            self.last_Z = Z
            off = 0
            offsets = []
            for b in self.batches:
                offsets.append(off)
                off += b["student_input_ids"].size(0)
            for bi in range(rank, len(self.batches), world):
                b = self.batches[bi]
                idx = list(range(offsets[bi], offsets[bi] + b["student_input_ids"].size(0)))
                for cond in CONDITIONS:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        terms = self._record_terms(b, self._cond_latents(Z, cond, idx), dev)
                    for rid, t in zip(b["record_ids"], terms):
                        recs.setdefault(rid, {})[cond] = t
            if world > 1:
                parts = [None] * world
                dist.all_gather_object(parts, recs)
                recs = {k: v for p in parts for k, v in p.items()}
        finally:
            self.model.train(was_training)
        res = summarize_records(recs)
        res.update({f"val/{k}": v for k, v in cross_question_cosine(Z).items()})
        lat_rms = float(Z.pow(2).mean().sqrt())
        tok_rms = float(self.model.get_input_embeddings().weight.float().pow(2).mean().sqrt())
        res["val/latent_rms_ratio"] = lat_rms / tok_rms
        if return_records:
            res["_records"] = recs
        return res

    def grad_split(self) -> Dict[str, float]:
        """Rank 0 only. autograd.grad: no .grad written, no optimizer step, no DDP reducer involvement."""
        dev = next(self.model.parameters()).device
        proj = list(self.model.latent_projector.parameters())
        was_training = self.model.training
        self.model.eval()
        b = self.grad_batch
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lat = self.model.forward_plan_distill_stage2(
                    input_ids=b["student_input_ids"].to(dev), attention_mask=b["student_attention_mask"].to(dev),
                    position_ids=b["student_position_ids"].to(dev), labels=b["student_labels"].to(dev),
                    boundary_positions=b["student_boundary"].to(dev), layer_ids=[self.model.config.num_hidden_layers],
                    latent_positions=b["latent_positions"].to(dev),
                    latent_start_positions=b["latent_start_positions"].to(dev), debug_latent_grad=True)
                z = lat["latent_embeddings"]
                zs = lat["projector_outputs"]     # z_1..z_L before insertion: ancestors of BOTH branches
                                                  # ("latent_embeddings" is a sibling index of the exec path)
                rec = self.model.forward_plan_reconstruction(z, b["planrec_input_ids"].to(dev),
                                                             b["planrec_attention_mask"].to(dev), b["planrec_labels"].to(dev))
                l_exec = lat["token_nll"].float().mean()
                l_rec = rec["token_nll"].float().mean()
            ge = torch.autograd.grad(l_exec, proj + list(zs), retain_graph=True, allow_unused=True)
            gr = torch.autograd.grad(l_rec, proj + list(zs), allow_unused=True)
        finally:
            self.model.train(was_training)
        n = len(proj)
        nrm = lambda gs: float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)))
        flat = lambda gs: torch.cat([g.float().flatten() for g in gs if g is not None])
        cos = lambda x, y: float(torch.nn.functional.cosine_similarity(flat(x), flat(y), dim=0))
        out = {"grad/projector_exec": nrm(ge[:n]), "grad/projector_planrec_unweighted": nrm(gr[:n]),
               "grad/projector_planrec_weighted": self.lambda_p * nrm(gr[:n]),
               "grad/z_exec": nrm(ge[n:]), "grad/z_planrec_weighted": self.lambda_p * nrm(gr[n:]),
               "grad/z1_exec": nrm(ge[n:n + 1]), "grad/z1_planrec_weighted": self.lambda_p * nrm(gr[n:n + 1]),
               "grad/cos_projector_planrec_vs_exec": cos(gr[:n], ge[:n]),
               "grad/cos_z_planrec_vs_exec": cos(gr[n:], ge[n:]),
               "grad/probe_l_exec": float(l_exec), "grad/probe_l_planrec": float(l_rec)}
        out["grad/projector_planrec_over_exec"] = out["grad/projector_planrec_weighted"] / max(out["grad/projector_exec"], 1e-30)
        return out


def _wmean(recs, cond, key):
    s = sum(r[cond][f"{key}_sum"] for r in recs.values())
    n = sum(r[cond][f"{key}_n"] for r in recs.values())
    return s / max(n, 1)


def _emean(recs, cond, key):
    vals = [r[cond][f"{key}_sum"] / r[cond][f"{key}_n"] for r in recs.values() if r[cond][f"{key}_n"] > 0]
    return sum(vals) / max(len(vals), 1)


def summarize_records(recs) -> Dict[str, float]:
    res = {}
    keys = {"exec": "exec_ce", "exec_nonrep": "exec_nonrep_ce", "rec": "planrec_ce", "rec_content": "planrec_content_ce",
            "rec_content32": "planrec_content32_ce", "rec_math": "planrec_math_ce"}
    for cond in CONDITIONS:
        for key, name in keys.items():
            res[f"val/{name}_{cond}"] = _wmean(recs, cond, key)
            res[f"val/{name}_{cond}_per_example"] = _emean(recs, cond, key)
    for name in keys.values():
        for cond in ("shuffled", "mean", "zero"):
            res[f"val/{name.replace('_ce', '')}_gain_{cond}"] = res[f"val/{name}_{cond}"] - res[f"val/{name}_learned"]
    res["val/n_records"] = len(recs)
    return res


def bootstrap_gain(recs, key: str, cond: str, n: int = 4000, seed: int = 0, per_example: bool = False):
    """Paired bootstrap over records of CE(cond) - CE(learned). Token-weighted (default) or per-example mean."""
    ids = sorted(recs)
    if per_example:
        ids = [i for i in ids if recs[i]["learned"][f"{key}_n"] > 0]
        d = [recs[i][cond][f"{key}_sum"] / recs[i][cond][f"{key}_n"] - recs[i]["learned"][f"{key}_sum"] / recs[i]["learned"][f"{key}_n"]
             for i in ids]
        w = [1.0] * len(ids)
    else:
        d = [(recs[i][cond][f"{key}_sum"] - recs[i]["learned"][f"{key}_sum"]) / max(recs[i]["learned"][f"{key}_n"], 1) for i in ids]
        w = [recs[i]["learned"][f"{key}_n"] for i in ids]
    stat = lambda ii: sum(d[i] * w[i] for i in ii) / max(sum(w[i] for i in ii), 1e-12)
    rng = random.Random(seed)
    idx = list(range(len(ids)))
    bs = sorted(stat([rng.choice(idx) for _ in idx]) for _ in range(n))
    return {"estimate": stat(idx), "ci95": [bs[int(0.025 * n)], bs[int(0.975 * n)]],
            "frac_records_positive": sum(x > 0 for x in d) / max(len(d), 1), "n_records": len(ids)}
