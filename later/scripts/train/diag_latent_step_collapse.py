"""Latent-step collapse diagnostic (forward-only: model.eval() + torch.inference_mode(), no optimizer, no backward).

Where does the cross-question collapse of the L=8 recurrent latents happen?
  h_in[t]  : input of LatentProjector at step t (= Qwen3Model.last_hidden_state, i.e. AFTER the final RMSNorm;
             t=1: hidden at latent_start = last token of '<EXPLORATION>\\n'; t>1: hidden at slot t-1)
  z_raw[t] : projector MLP output before the output_rms rescale
  z_out[t] : projector output after output_rms = the embedding actually written into slot t
  h_out8   : hidden after z_8 (not fed to the projector; reference only)
Captured with read-only forward hooks on model.latent_projector / .net / .model inside the training builder
(_build_latent_recursive_embeds, the same path used by forward_plan_distill_stage2 and the training probe).

P0 gate : captured z_out == builder output; h_in[t+1] == hidden after z_t; batch-1 vs batch-8 extraction;
          execution CE / target logits with vs without hooks; repeat-run determinism.
Test A  : per step, for h_in / z_raw / z_out: cross-question cosine (mean/median/P10/P90), relative cross-question
          std, mean pairwise distance / mean norm, centred spectrum (effective rank, top-k variance share),
          linear CKA with step 1 and between h_in and z_out (same step); within-question step-to-step similarity.
          Reference (same space, same records): explicit-path hidden states / gold token embeddings at 8 evenly
          spaced plan-body positions (Stage-2 model reading the gold plan).
Test B  : controlled forced forward via latent_override (all 8 slots fixed, NO recursive recompute after a swap):
          swap single steps / prefixes / suffixes with a donor question (fixed derangement); execution CE deltas.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from later.src.train.stage2_common import load_stage2_model, load_tokenizer
from later.src.train.stage2_plan_dataset import PLAN_HEADER, Stage2Collator, build_views, stage2_tokens
from later.src.train.stage2_planrec_monitor import fixed_derangement

STEPS_MAIN = [1, 2, 4, 8]


class Capture:
    """Read-only hooks; they return None so module outputs are untouched."""

    def __init__(self, model):
        self.model, self.h_in, self.z_raw, self.z_out, self.model_out, self.handles = model, [], [], [], [], []

    def __enter__(self):
        p = self.model.latent_projector
        self.handles = [
            p.register_forward_hook(lambda m, a, o: (self.h_in.append(a[0].detach().float().clone()),
                                                     self.z_out.append(o.detach().float().clone())) and None),
            p.net.register_forward_hook(lambda m, a, o: self.z_raw.append(o.detach().float().clone()) and None),
            self.model.model.register_forward_hook(
                lambda m, a, o: self.model_out.append(o.last_hidden_state[:, -1].detach().float().clone()) and None),
        ]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


def builder(model, b):
    dev = "cuda"
    lp, ls = b["latent_positions"].to(dev), b["latent_start_positions"].to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        over, _, stats = model._build_latent_recursive_embeds(
            input_ids=b["student_input_ids"].to(dev), attention_mask=b["student_attention_mask"].to(dev),
            position_ids=b["student_position_ids"].to(dev), cot_mask=None, latent_positions=lp, latent_slot_mask=None,
            latent_start_positions=ls, global_prompt_len=int(ls.max()) + 1, use_no_grad=True)
    rows = torch.arange(lp.size(0), device=dev).unsqueeze(1)
    return over[rows, lp].float(), stats


def extract(model, b, L):
    with Capture(model) as c:
        Z, stats = builder(model, b)
    assert len(c.h_in) == L and len(c.model_out) == L + 1, (len(c.h_in), len(c.model_out))
    st = lambda xs: torch.stack(xs, 1)  # [B, L, d]
    return {"h_in": st(c.h_in), "z_raw": st(c.z_raw), "z_out": st(c.z_out), "h_after": st(c.model_out[1:]),
            "Z_builder": Z, "stats": stats}


def exec_forward(model, b, latent_override=None):
    dev = "cuda"
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.forward_plan_distill_stage2(
            input_ids=b["student_input_ids"].to(dev), attention_mask=b["student_attention_mask"].to(dev),
            position_ids=b["student_position_ids"].to(dev), labels=b["student_labels"].to(dev),
            boundary_positions=b["student_boundary"].to(dev), layer_ids=[36],
            latent_positions=b["latent_positions"].to(dev), latent_start_positions=b["latent_start_positions"].to(dev),
            return_target_logits=True, latent_override=latent_override)
    return out


# ---------------------------------------------------------------- metrics
def pair_stats(v):
    v = np.asarray(v)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90))}


def lin_cka(X, Y):
    X, Y = X - X.mean(0), Y - Y.mean(0)
    xy = np.linalg.norm(X.T @ Y) ** 2
    return float(xy / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y) + 1e-30))


def step_metrics(X):
    """X [N, d] float64 (one step, one representation)."""
    N = X.shape[0]
    off = ~np.eye(N, dtype=bool)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
    cos = (Xn @ Xn.T)[off]
    mu = X.mean(0)
    C = X - mu
    rel_std = math.sqrt((C ** 2).mean()) / (math.sqrt((X ** 2).mean()) + 1e-12)
    norms = np.linalg.norm(X, axis=1)
    sq = (X ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))[off]
    s = np.linalg.svd(C, compute_uv=False)
    var = s ** 2 / (s ** 2).sum()
    erank = float(np.exp(-(var[var > 0] * np.log(var[var > 0])).sum()))
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-30)
    return {"cos_raw": pair_stats(cos), "cos_centered_mean": float((Cn @ Cn.T)[off].mean()),
            "relative_cross_question_std": rel_std, "pairwise_dist_over_mean_norm": float(D.mean() / norms.mean()),
            "common_component_norm_over_mean_norm": float(np.linalg.norm(mu) / norms.mean()),
            "centered_effective_rank": erank, "participation_ratio": float(1.0 / (var ** 2).sum()),
            "top1_var_share": float(var[0]), "top5_var_share": float(var[:5].sum()),
            "mean_l2_norm": float(norms.mean()), "mean_rms": float(np.sqrt((X ** 2).mean(1)).mean())}


def nn_retrieval(A, B):
    """Top-1 accuracy of matching question i's centred A-vector to its own centred B-vector (cosine)."""
    A, B = A - A.mean(0), B - B.mean(0)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-30
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-30
    return float((np.argmax(A @ B.T, 1) == np.arange(len(A))).mean())


def boot_mean(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d)
    bs = np.sort(rng.choice(d, (n, len(d))).mean(1))
    return {"estimate": float(d.mean()), "ci95": [float(bs[int(0.025 * n)]), float(bs[int(0.975 * n) - 1])],
            "frac_positive": float((d > 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip_b", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    log = open(os.path.join(a.out, "logs.txt"), "a")

    def P(*x):
        s = " ".join(str(y) for y in x)
        print(s, flush=True)
        log.write(f"[{time.time() - t0:7.1f}s] {s}\n")
        log.flush()
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    tok = load_tokenizer(a.model)
    T = stage2_tokens(tok)
    val = pd.read_parquet(f"{a.data_dir}/stage2_val.parquet")
    rows = [val.iloc[i].to_dict() for i in range(a.n)]
    P("records", a.n, "distinct questions", len({r["question"] for r in rows}))
    model = load_stage2_model(a.model, latent_embed_builder_no_grad=True, latent_projector_dropout=0.0,
                              dtype=torch.float32).cuda().eval()
    L = int(json.load(open(os.path.join(a.model, "stage2_meta.json")))["latent_steps"])
    drop = [m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    P("loaded", a.model, "output_rms", model.latent_projector.output_rms, "training", model.training,
      "dropout p values", sorted(set(drop)), "attn_dropout", getattr(model.config, "attention_dropout", None))
    col = Stage2Collator(tok.pad_token_id)
    views = [build_views(tok, r, L, T, 4096) for r in rows]
    batches = [col(views[s: s + a.bs]) for s in range(0, a.n, a.bs)]
    gate = {}
    with torch.inference_mode():
        # ---------------- P0 gate
        b0 = batches[0]
        ref = exec_forward(model, b0)
        with Capture(model) as c:
            hooked = exec_forward(model, b0)
        gate["hook_calls_during_exec_forward"] = {"projector": len(c.h_in), "model": len(c.model_out)}
        ref2 = exec_forward(model, b0)
        gate["hooks_vs_no_hooks_max_abs_token_nll"] = float((ref["token_nll"] - hooked["token_nll"]).abs().max())
        gate["hooks_vs_no_hooks_max_abs_logit"] = float((ref["target_logits"] - hooked["target_logits"]).abs().max())
        gate["repeat_run_max_abs_logit"] = float((ref["target_logits"] - ref2["target_logits"]).abs().max())
        gate["exec_ce_no_hooks"] = float(ref["token_nll"].mean())
        gate["exec_ce_with_hooks"] = float(hooked["token_nll"].mean())
        # extraction
        E = [extract(model, b, L) for b in batches]
        cat = lambda k: torch.cat([e[k] for e in E])
        H, ZR, ZO, HA, ZB = cat("h_in"), cat("z_raw"), cat("z_out"), cat("h_after"), cat("Z_builder")
        gate["builder_stats"] = E[0]["stats"]
        gate["z_out_vs_builder_written_max_abs"] = float((ZO - ZB).abs().max())
        gate["h_in_next_vs_h_after_max_abs"] = float((H[:, 1:] - HA[:, :-1]).abs().max())
        rms_out = ZO.pow(2).mean(-1).sqrt()
        gate["z_out_rms_min_max"] = [float(rms_out.min()), float(rms_out.max())]
        gate["output_rms_config"] = float(model.latent_projector.output_rms or 0)
        gate["z_out_equals_rescaled_z_raw_max_abs"] = float(
            (ZR * (model.latent_projector.output_rms / ZR.pow(2).mean(-1, keepdim=True).add(1e-12).sqrt()) - ZO).abs().max())
        # override with learned Z reproduces the builder path
        ov = exec_forward(model, b0, latent_override=ZB[: b0["student_input_ids"].size(0)])
        gate["override_learned_vs_builder_max_abs_logit"] = float((ov["target_logits"] - ref["target_logits"]).abs().max())
        gate["override_learned_vs_builder_ce_diff"] = float(ov["token_nll"].mean() - ref["token_nll"].mean())
        # batch-1 vs batch-8 extraction (padding / per-row positions)
        e1 = [extract(model, col([v]), L) for v in views[: a.bs]]
        H1 = torch.cat([e["h_in"] for e in e1])
        Z1 = torch.cat([e["z_out"] for e in e1])
        cosf = lambda x, y: torch.nn.functional.cosine_similarity(x, y, dim=-1)
        gate["batch1_vs_batch8_h_in_min_cos"] = float(cosf(H1, H[: a.bs]).min())
        gate["batch1_vs_batch8_z_out_min_cos"] = float(cosf(Z1, ZO[: a.bs]).min())
        gate["batch1_vs_batch8_z_out_max_abs_over_rms"] = float((Z1 - ZO[: a.bs]).abs().max() / ZO.pow(2).mean().sqrt())
        # numerical-noise floor (batch shape / bf16 kernels) relative to the cross-question spread, per step
        for nm, X1, Xall in (("h_in", H1, H), ("z_out", Z1, ZO)):
            noise = (X1 - Xall[: a.bs]).pow(2).mean((0, 2)).sqrt()
            spread = (Xall - Xall.mean(0)).pow(2).mean((0, 2)).sqrt()
            gate[f"batch_noise_rms_over_cross_question_std_{nm}"] = [round(float(x), 4) for x in noise / spread]
        e_rep = extract(model, b0, L)
        gate["repeat_extraction_max_abs_z_out"] = float((e_rep["z_out"] - E[0]["z_out"]).abs().max())
        P("GATE", json.dumps(gate))
        # ---------------- reference: explicit path reading the gold plan (same records)
        ref_h, ref_e = [], []
        hdr_ids = len(tok(PLAN_HEADER + "\n", add_special_tokens=False)["input_ids"])
        for s in range(0, a.n, a.bs):
            bb = batches[s // a.bs]
            ids = bb["explicit_input_ids"].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hs = model.model(input_ids=ids, attention_mask=bb["explicit_attention_mask"].cuda(),
                                 position_ids=bb["explicit_position_ids"].cuda(), use_cache=False).last_hidden_state.float()
            emb = model.get_input_embeddings()(ids).float()
            for i, v in enumerate(views[s: s + a.bs]):
                p0 = v["latent_start"] + 1                       # first plan token in the explicit sequence
                n_plan = sum(1 for x in v["planrec_labels"] if x != -100)
                body0 = p0 + hdr_ids
                n_body = max(1, n_plan - hdr_ids - 1)
                pos = [body0 + int(k * n_body / L) for k in range(L)]
                ref_h.append(hs[i, pos])
                ref_e.append(emb[i, pos])
        RH, RE = torch.stack(ref_h), torch.stack(ref_e)
        # ---------------- Test B (controlled forced forward)
        swap = {}
        if not a.skip_b:
            perm = fixed_derangement(a.n, seed=0)
            conds = {"original": []}
            for t in STEPS_MAIN:
                conds[f"swap_step_{t}"] = [t]
            for t in (1, 2, 4):
                conds[f"swap_prefix_1:{t}"] = list(range(1, t + 1))
            for t in (2, 4, 8):
                conds[f"swap_suffix_{t}:8"] = list(range(t, L + 1))
            conds["swap_all_1:8"] = list(range(1, L + 1))
            per = {c: [] for c in conds}
            per16 = {c: [] for c in conds}
            perfull = {c: [] for c in conds}
            kl0 = {c: [] for c in conds}
            for bi, bb in enumerate(batches):
                B = bb["student_input_ids"].size(0)
                idx = list(range(bi * a.bs, bi * a.bs + B))
                rep = bb["student_seg_R"][:, 1:].cuda()
                base_logits = None
                for c, steps in conds.items():
                    z = ZB[idx].clone()
                    for t in steps:
                        z[:, t - 1] = ZB[[perm[i] for i in idx], t - 1]
                    out = exec_forward(model, bb, latent_override=z)
                    sel = out["target_mask"]
                    nll = out["token_nll"].double()
                    nonrep = (~rep)[sel]
                    lg = torch.log_softmax(out["target_logits"], -1)
                    cnt = sel.sum(1).tolist()
                    o = 0
                    if c == "original":
                        base_logits = lg
                    for i in range(B):
                        sl = slice(o, o + cnt[i])
                        nr = nonrep[sl]
                        v = nll[sl][nr]
                        per[c].append(float(v.mean()))
                        per16[c].append(float(v[:16].mean()))
                        perfull[c].append(float(nll[sl].mean()))
                        p, q = base_logits[o], lg[o]
                        kl0[c].append(float((p.exp() * (p - q)).sum()))
                        o += cnt[i]
                P("TestB batch", bi, "done")
            for c in conds:
                swap[c] = {"steps_replaced": conds[c], "ce_nonrep_mean": float(np.mean(per[c])),
                           "ce_full_mean": float(np.mean(perfull[c])), "ce_first16_nonrep_mean": float(np.mean(per16[c])),
                           "delta_ce_nonrep_vs_original": boot_mean(np.array(per[c]) - np.array(per["original"])),
                           "delta_ce_first16_vs_original": boot_mean(np.array(per16[c]) - np.array(per16["original"])),
                           "first_token_kl_vs_original_mean": float(np.mean(kl0[c])),
                           "first_token_kl_vs_original_max": float(np.max(kl0[c]))}
            swap["_per_record"] = {rows[i]["record_id"]: {"donor": rows[perm[i]]["record_id"],
                                                          **{c: {"ce_nonrep": per[c][i], "ce_first16": per16[c][i],
                                                                 "kl0": kl0[c][i]} for c in conds}} for i in range(a.n)}
            swap["_protocol"] = ("latent_override: all 8 slot vectors fixed (forced forward), the recursion is NOT rerun "
                                 "after a swap; donor = fixed derangement seed 0 over the 128 records; per-question mean "
                                 "over non-repeated execution tokens (student_seg_R excluded); first-token KL = "
                                 "KL(original || swapped) at the first execution target")
            json.dump(swap, open(os.path.join(a.out, "step_swap_metrics.json"), "w"), indent=1)
    # ---------------- Test A metrics
    reps = {"h_in": H, "z_raw": ZR, "z_out": ZO}
    np_ = {k: v.double().cpu().numpy() for k, v in reps.items()}
    ref_np = {"ref_explicit_plan_hidden": RH.double().cpu().numpy(), "ref_explicit_plan_token_embed": RE.double().cpu().numpy()}
    h_out8 = HA[:, -1].double().cpu().numpy()
    rows_csv, js = [], {"per_step": {}, "within_question": {}, "reference": {}}
    for rep, X in {**np_, **ref_np}.items():
        for t in range(1, L + 1):
            m = step_metrics(X[:, t - 1])
            m["cka_with_step1"] = lin_cka(X[:, t - 1], X[:, 0])
            m["nn_retrieval_vs_step1"] = nn_retrieval(X[:, t - 1], X[:, 0])
            if rep in ("z_out", "z_raw"):
                m["cka_with_h_in_same_step"] = lin_cka(X[:, t - 1], np_["h_in"][:, t - 1])
                m["nn_retrieval_vs_h_in_same_step"] = nn_retrieval(X[:, t - 1], np_["h_in"][:, t - 1])
            (js["reference"] if rep.startswith("ref") else js["per_step"]).setdefault(rep, {})[t] = m
            flat = {"representation": rep, "step": t}
            for k, v in m.items():
                if isinstance(v, dict):
                    flat.update({f"{k}_{kk}": vv for kk, vv in v.items()})
                else:
                    flat[k] = v
            rows_csv.append(flat)
    m8 = step_metrics(h_out8)
    m8["cka_with_h_in_step1"] = lin_cka(h_out8, np_["h_in"][:, 0])
    js["per_step"]["h_after_z8"] = m8
    rows_csv.append({"representation": "h_after_z8", "step": 9,
                     **{k: v for k, v in m8.items() if not isinstance(v, dict)},
                     **{f"cos_raw_{k}": v for k, v in m8["cos_raw"].items()}})
    for rep, X in np_.items():
        wq = {}
        for t in range(1, L):
            A_, B_ = X[:, t - 1], X[:, t]
            c = (A_ * B_).sum(1) / (np.linalg.norm(A_, axis=1) * np.linalg.norm(B_, axis=1))
            d = np.linalg.norm(A_ - B_, axis=1) / np.linalg.norm(A_, axis=1)
            Ac, Bc = A_ - A_.mean(0), B_ - B_.mean(0)
            cc = (Ac * Bc).sum(1) / (np.linalg.norm(Ac, axis=1) * np.linalg.norm(Bc, axis=1))
            wq[f"{t}->{t + 1}"] = {"cos_mean": float(c.mean()), "rel_l2_step_mean": float(d.mean()),
                                   "centered_cos_mean": float(cc.mean())}
        js["within_question"][rep] = wq
    pd.DataFrame(rows_csv).to_csv(os.path.join(a.out, "per_step_metrics.csv"), index=False)
    js["gate"] = gate
    js["record_ids"] = [r["record_id"] for r in rows]
    js["n_distinct_questions"] = len({r["question"] for r in rows})
    json.dump(js, open(os.path.join(a.out, "per_step_metrics.json"), "w"), indent=1)
    torch.save({"h_in": H.half().cpu(), "z_out": ZO.half().cpu(), "z_raw": ZR.half().cpu(), "h_after": HA.half().cpu(),
                "record_ids": js["record_ids"]}, os.path.join(a.out, "latents_fp16.pt"))
    P("DONE in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
