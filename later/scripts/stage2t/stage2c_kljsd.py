"""Stage 2C GPU0 / GPU1 controlled comparison: LaTER-style forward KL vs plan-conditioned JSD (both with the free-running Z_S), <= 200 local steps
from Stage 2B joint global step 500.  The ONLY difference between the two runs is --distill_type.

  L = 1.0 * KD + 1.0 * Anchor + 0.1 * Exec_FR + 1.0 * Cos_TF
  KD      = mean_t D(q_t, p_t),  q_t = sg p_T(.|Q + P_text, 8 ZERO slots, y<t)   (frozen step-500 copy; no answer hint, never sees Z_S / Z_T)
                                 p_t = p_S(.|Q, Z_S^FR, y<t)                       (Z_S graph kept -> projector + LoRA)
            forward_kl: D = KL(q || p) = sum_v q (log q - log p)      jsd: D = JSD(q, p)      (fp32, full vocab, T = 1, E-token mask)
  Anchor  = - mean_t JSD(p_t, sg p_S(.|Q, zeros, y<t))                (already carries the minus)
  Exec_FR = CE over E + '</EXECUTION><|im_end|>' on the same with-branch forward / same Z_S
  Cos_TF  = joint_2b.cosd(joint_2b.tf_latents(S, Q, Z_T), Z_T)
Optimizer / data identical to the earlier four-loss runs (AdamW lr 1e-4 wd 0 constant, clip 1.0, bf16 autocast, seed 0, 16/step, batches(seed=42)).
Modes: audit (Gate 0A teacher plan margins + Gate 0B divergence unit tests / per-term grads / leak / alignment; no optimizer step) | train.
"""
import argparse
import hashlib
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from later.scripts.bridge.mini_bridge import correct, last_boxed
from later.scripts.stage2s.stochastic_pilot import grad_norm
from later.scripts.stage2t.anchorsim_2c import fwd, jsd, nll, student_pre
from later.scripts.stage2t.contrastive_2b import generate_batch
from later.scripts.stage2t.joint_2b import cosd, tf_latents
from later.scripts.stage2t.joint_2b_continue import diag
from later.scripts.stage2t.plan_teacher_gate0 import PLAN_HINT, plan_teacher_pre
from later.scripts.stage2t.stage2c_long import INIT_GLOBAL_STEP, degenerate, sha256
from later.scripts.stage2t.teacher_plan_audit import INVALID_WHY, SPEC
from later.scripts.stage2t.three_step import TStudent, batches, load_jsonl, save_ck

LAMS = {"kd": 1.0, "anchor": 1.0, "exec": 0.1, "cos": 1.0}
DIV = None
P_TARGET = "A"            # pre-registered (before any training): the single target plan of each held-out pair for FR decision metrics / adherence
SHUF_SEEDS = (0, 1, 2)
SAFETY = {"smoke": "SAFETY_FAIL if (acc drop >= 5/16 and format drop >= 6/16) or >= 3/4 fixed generations degenerate/format-broken "
                   "or (gold NLL >= step0 + 0.20 and acc drop >= 4/16)  [vs this run's own step 0, same 16 questions]",
          "light": "SAFETY_FAIL if >= 3/4 fixed generations degenerate/format-broken or (16-q gold NLL >= step0 + 0.20 and 4-gen correct <= step0_first4 - 2)"}


def kl_fwd(lt, ls):
    """[T,V] teacher logits, [T,V] student logits -> [T] KL(p_T || p_S), fp32 full vocab; gradient through whichever input is not detached."""
    lq, lp = F.log_softmax(lt.float(), -1), F.log_softmax(ls.float(), -1)
    return (lq.exp() * (lq - lp)).sum(-1)


def kd_fn(lt, ls):
    return kl_fwd(lt.detach(), ls) if DIV == "forward_kl" else jsd(lt.detach(), ls)


def tpre(Tm, q, plan):
    return plan_teacher_pre(Tm, q, plan)


def item(S, Tm, s, n_tail):
    q, tgt = s["question"], S.target_ids(s["execution"]); nE = len(tgt) - n_tail; pre = student_pre(S, q)
    z = S.zlat(q)                                                     # free-running Z_S, graph kept
    lw_full, yw, _ = fwd(S, pre, z, tgt, len(tgt)); lw = lw_full[:nE]
    with torch.no_grad():
        lb, yb, _ = fwd(S, pre, torch.zeros_like(z), tgt, nE)
        lt, yt, _ = fwd(Tm, tpre(Tm, q, s["plan_norm"]), torch.zeros_like(z), tgt, nE)
    T = S.teacher(s["plan_norm"])
    parts = {"kd": kd_fn(lt, lw).mean(), "anchor": -jsd(lw, lb.detach()).mean(),
             "exec": F.cross_entropy(lw_full, torch.tensor(tgt, device=S.dev)), "cos": cosd(tf_latents(S, q, T), T)}
    info = {"nE": nE, "n_tgt": len(tgt), "aligned": bool(torch.equal(yw[:nE], yb) and torch.equal(yb, yt)) and yw.tolist() == tgt, "lt": lt, "lw": lw.detach(), "z": z}
    return parts, info


def total(parts):
    return sum(LAMS[k] * parts[k] for k in LAMS)


def derange(n, seed):
    r = random.Random(1000 + seed)
    while True:
        p = list(range(n)); r.shuffle(p)
        if all(p[i] != i for i in range(n)):
            return p


def flat_grad(ps):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().float() for p in ps])


def term_grads(S, Tm, B, n_tail):
    """per-term weighted grads on fixed samples B (separate backward per term), norms + cosine of KD vs the others, per module. Leaves grads cleared."""
    params = S.params(); P = [p for _, p in S.proj]; Lo = [p for _, p in S.lora]; G, out = {}, {}
    for nm in ["kd", "anchor", "exec", "cos", "total"]:
        for p in params:
            p.grad = None
        for s in B:
            parts, _ = item(S, Tm, s, n_tail)
            ((total(parts) if nm == "total" else LAMS[nm] * parts[nm]) / len(B)).backward()
        G[nm] = (flat_grad(P), flat_grad(Lo))
        out[nm] = {"projector": float(G[nm][0].norm()), "lora": float(G[nm][1].norm()), "finite": bool(torch.isfinite(G[nm][0]).all() and torch.isfinite(G[nm][1]).all())}
    for o in ["anchor", "exec", "cos"]:
        for j, mod in enumerate(("projector", "lora")):
            out[f"gradcos_kd_{o}_{mod}"] = float(F.cosine_similarity(G["kd"][j], G[o][j], dim=0))
    out["teacher_grads_none"] = all(p.grad is None for p in Tm.peft.parameters())
    out["frozen_student_grads_none"] = all(p.grad is None for p in S.peft.parameters() if not p.requires_grad)
    for p in params:
        p.grad = None
    return out


def pair_specs(s2b, tok):
    out = []
    for p in [p for p in load_jsonl(f"{s2b}/data/pairs.jsonl") if p["role"] == "heldout8"]:
        qid = p["id"][:11]; sp = SPEC[qid]
        if sp is None:
            out.append({"p": p, "valid": False, "why": INVALID_WHY[qid]}); continue
        c, UA, UB = sp(p["execution_A"], p["execution_B"])
        ia, ib = tok.encode(c + UA, add_special_tokens=False), tok.encode(c + UB, add_special_tokens=False)
        k = 0
        while k < min(len(ia), len(ib)) and ia[k] == ib[k]:
            k += 1
        out.append({"p": p, "valid": True, "c": c, "UA": UA, "UB": UB, "ia": ia, "ib": ib, "k": k})
    return out


def span_lp(lg, ids, k):
    l = F.log_softmax(lg.float(), -1)[k:].gather(1, torch.tensor(ids[k:], device=lg.device).unsqueeze(1))[:, 0]
    return float(l.mean()), float(l.sum()), len(ids) - k


@torch.no_grad()
def decision_eval(S, Tm, PS):
    """7 valid held-out pairs.  ORACLE: M_A / M_B with frozen 2A Z_A^T / Z_B^T (student prompt, no plan).  FR: Z_S vs teacher(Q+P_target, zeros)
    at the decision row k; decision NLL of U_target; dNLL vs zero / 3 shuffled Z_S (derangements within the valid pairs)."""
    V = [x for x in PS if x["valid"]]; Zs = [S.zlat(x["p"]["question"]) for x in V]; rows = []
    for i, x in enumerate(V):
        p, q = x["p"], x["p"]["question"]; pre = student_pre(S, q); k = x["k"]
        ZA, ZB = S.teacher(p["plan_A_norm"]), S.teacher(p["plan_B_norm"])
        s = {}
        for zn, z in (("ZA", ZA), ("ZB", ZB), ("ZS", Zs[i]), ("Z0", torch.zeros_like(Zs[i]))):
            for un, ids in (("A", x["ia"]), ("B", x["ib"])):
                lg, y, _ = fwd(S, pre, z, ids, len(ids)); assert y.tolist() == ids
                s[(zn, un)] = span_lp(lg, ids, k)
                if zn == "ZS" and un == P_TARGET:
                    dec_s = lg[k]
        tg = x["ia"] if P_TARGET == "A" else x["ib"]; plan_t = p["plan_A_norm"] if P_TARGET == "A" else p["plan_B_norm"]
        lt, yt, _ = fwd(Tm, tpre(Tm, q, plan_t), torch.zeros_like(Zs[i]), tg, len(tg)); assert yt.tolist() == tg
        sh = []
        for sd in SHUF_SEEDS:
            j = derange(len(V), sd)[i]; lg, _, _ = fwd(S, pre, Zs[j], tg, len(tg)); sh.append(span_lp(lg, tg, k)[0])
        mA = s[("ZA", "A")][0] - s[("ZB", "A")][0]; mB = s[("ZB", "B")][0] - s[("ZA", "B")][0]
        st = s[("ZS", P_TARGET)][0]
        rows.append({"id": p["id"], "k": k, "len_A": len(x["ia"]) - k, "len_B": len(x["ib"]) - k, "margin_A": mA, "margin_B": mB, "two_sided_margin_pass": mA > 0 and mB > 0,
                     "s": {f"{u}|{zn}": v[0] for (zn, u), v in s.items()}, "s_total": {f"{u}|{zn}": v[1] for (zn, u), v in s.items()},
                     "decision_kl": float(kl_fwd(lt[k:k + 1], dec_s.unsqueeze(0))[0]), "decision_jsd": float(jsd(lt[k:k + 1], dec_s.unsqueeze(0))[0]),
                     "decision_nll": -st, "dnll_zero_decision": st - s[("Z0", P_TARGET)][0], "dnll_shuffle_decision": st - float(np.mean(sh)),
                     "fr_pref_target_minus_other": st - s[("ZS", "B" if P_TARGET == "A" else "A")][0],
                     "teacher_decision_top1": S.tok.decode([int(lt[k].argmax())]), "student_decision_top1": S.tok.decode([int(dec_s.argmax())])})
    agg = {"n_valid": len(rows), "oracle_two_sided_margin_pass": sum(r["two_sided_margin_pass"] for r in rows),
           **{k: float(np.mean([r[k] for r in rows])) for k in ("margin_A", "margin_B", "decision_kl", "decision_jsd", "decision_nll", "dnll_zero_decision",
                                                               "dnll_shuffle_decision", "fr_pref_target_minus_other")}}
    return agg, rows


@torch.no_grad()
def tf_eval(S, Tm, ev, n_tail):
    """16 held-out val questions, gold prefix: forward KL / JSD vs the plan teacher, gold NLL, with/blind JSD, dNLL zero / shuffle (3 derangements)."""
    Z = [S.zlat(s["question"]) for s in ev]; rows = []; perms = [derange(len(ev), sd) for sd in SHUF_SEEDS]
    for i, s in enumerate(ev):
        q, tgt = s["question"], S.target_ids(s["execution"]); nE = len(tgt) - n_tail; pre = student_pre(S, q); y = torch.tensor(tgt[:nE], device=S.dev)
        lw, _, _ = fwd(S, pre, Z[i], tgt, nE); lb, _, _ = fwd(S, pre, torch.zeros_like(Z[i]), tgt, nE)
        lt, _, _ = fwd(Tm, tpre(Tm, q, s["plan_norm"]), torch.zeros_like(Z[i]), tgt, nE)
        kl, js = kl_fwd(lt, lw), jsd(lt, lw); nw = nll(lw, y)
        sh = [nll(fwd(S, pre, Z[pm[i]], tgt, nE)[0], y) for pm in perms]
        rows.append({"id": s["id"], "nE": nE, "eval_kl_forward": float(kl.mean()), "eval_jsd": float(js.mean()), "eval_gold_nll": nw, "kl_sum": float(kl.sum()), "jsd_sum": float(js.sum()),
                     "jsd_with_blind": float(jsd(lw, lb).mean()), "delta_nll_zero_full": nll(lb, y) - nw, "delta_nll_shuffle_full": float(np.mean(sh)) - nw})
    agg = {k: float(np.mean([r[k] for r in rows])) for k in ("eval_kl_forward", "eval_jsd", "eval_gold_nll", "jsd_with_blind", "delta_nll_zero_full", "delta_nll_shuffle_full")}
    nt = sum(r["nE"] for r in rows)
    agg.update({"eval_kl_forward_tokw": sum(r["kl_sum"] for r in rows) / nt, "eval_jsd_tokw": sum(r["jsd_sum"] for r in rows) / nt, "n_q": len(rows), "n_tok": nt,
                "dnll_zero_frac_pos": float(np.mean([r["delta_nll_zero_full"] > 0 for r in rows])), "dnll_shuffle_frac_pos": float(np.mean([r["delta_nll_shuffle_full"] > 0 for r in rows]))})
    return agg, rows, Z


def gen_rows(S, ev, Z, gen_tokens):
    gens = generate_batch(S, [s["question"] for s in ev], torch.stack(Z), gen_tokens); rows = []
    for s, g in zip(ev, gens):
        p = last_boxed(g)
        rows.append({"id": s["id"], "pred": p, "correct": correct(p, str(s["answer"])), "format_ok": "</EXECUTION>" in g and "<|im_end|>" in g,
                     "empty": len(g.replace("</EXECUTION>", "").replace("<|im_end|>", "").strip()) < 20, "degenerate": degenerate(g), "text": g})
    return rows


def common_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def main():
    global DIV
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["audit", "train"], required=True); ap.add_argument("--distill_type", choices=["forward_kl", "jsd"], required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--s2b", required=True); ap.add_argument("--out", required=True); ap.add_argument("--init", required=True)
    ap.add_argument("--steps", type=int, default=200); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--n_single", type=int, default=16)
    ap.add_argument("--save_at", default="0,10,50,100,150,200"); ap.add_argument("--smoke_at", default="0,5,10"); ap.add_argument("--light_at", default="50,100,150,200")
    ap.add_argument("--grad_at", default="0,5,10,100,200"); ap.add_argument("--interv_at", default="0,10,200"); ap.add_argument("--diag_at", default="0,100")
    ap.add_argument("--gen_tokens", type=int, default=2048); ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--lam_kd", type=float, default=1.0, help="KD coefficient (user-set 0.25 for the GPU0 forward-KL run on 2026-10-14)")
    a = ap.parse_args(); DIV = a.distill_type; LAMS["kd"] = a.lam_kd
    for d in ("checkpoints", "evaluations"):
        os.makedirs(f"{a.out}/{d}", exist_ok=True)
    torch.manual_seed(0)
    ck = torch.load(a.init, weights_only=False)
    assert ck["step"] == INIT_GLOBAL_STEP and ck.get("phase") == "joint", "init must be Stage 2B joint global step 500"
    S = TStudent(a.model, a.s2b, seed=0); S.load(ck["state"])
    Tm = TStudent(a.model, a.s2b, seed=0); Tm.load(ck["state"])
    for p in Tm.peft.parameters():
        p.requires_grad_(False)
    Tm.peft.eval()
    t_snap = {n: p.detach().clone() for n, p in Tm.peft.named_parameters() if "lora_" in n or "latent_projector" in n}
    teacher_unchanged = lambda: all(torch.equal(p, t_snap[n]) for n, p in Tm.peft.named_parameters() if n in t_snap)
    n_tail = len(S.tok.encode("</EXECUTION><|im_end|>", add_special_tokens=False))
    singles = load_jsonl(f"{a.s2b}/data/train_singles.jsonl"); vals = load_jsonl(f"{a.s2b}/data/val_singles.jsonl")
    ev = vals[48:64]; PS = pair_specs(a.s2b, S.tok); B4 = singles[:4]
    params = S.params()
    with torch.no_grad():
        z4 = torch.stack([S.zlat(s["question"]) for s in B4]).float().cpu()
    sh = hashlib.sha256()
    for n, p in sorted(S.peft.named_parameters()):
        if p.requires_grad:
            sh.update(n.encode()); sh.update(p.detach().float().cpu().numpy().tobytes())
    meta = {"mode": a.mode, "distill_type": DIV, "init": a.init, "init_sha256": sha256(a.init), "init_global_step": ck["step"], "init_phase": ck.get("phase"),
            "gpu": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "SLURM_STEP_GPUS": os.environ.get("SLURM_STEP_GPUS"), "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
                    "name": torch.cuda.get_device_name(0), "uuid": str(torch.cuda.get_device_properties(0).uuid)},
            "lambdas": LAMS, "kd": "KL(sg p_T || p_S)" if DIV == "forward_kl" else "JSD(sg p_T, p_S)", "kd_space": "fp32 log_softmax, full vocab, T=1, E tokens (no close/eos tail)",
            "anchor": "-JSD(p_with, sg p_blind[zeros in 8 slots])", "exec": "CE over E + </EXECUTION><|im_end|>, same with-branch forward / same Z_S", "cos": "joint_2b TF cos vs frozen 2A Z_T",
            "teacher": "frozen copy of init; prompt = Q + PLAN_HINT(plan_norm) + 8 ZERO slots; no answer hint; never Z_S/Z_T", "plan_hint_template": PLAN_HINT,
            "student_input": "Q + Z_S^FR (8 free-running latents, graph kept)", "ema": False, "p_target_heldout": P_TARGET, "shuffle_seeds": SHUF_SEEDS, "safety_rules": SAFETY,
            "optimizer": f"AdamW(lr={a.lr}, wd=0, fresh), constant lr, no scheduler, clip 1.0", "precision": "bf16 autocast (no GradScaler), fp32 KD / JSD / CE / cos", "seed": 0,
            "batch": f"{a.n_single}/step, per-sample backward /{a.n_single}", "data": f"{a.s2b}/data/train_singles.jsonl", "data_sha256": sha256(f"{a.s2b}/data/train_singles.jsonl"),
            "n_train": len(singles), "data_order": f"three_step.batches(seed={a.data_seed})", "smoke_eval_ids": [s["id"] for s in ev], "light_gen_ids": [s["id"] for s in ev[:4]],
            "grad_audit_ids": [s["id"] for s in B4], "student_trainable": {"projector": sum(p.numel() for _, p in S.proj), "lora": sum(p.numel() for _, p in S.lora),
                                                                          "n_tensors": [len(S.proj), len(S.lora)], "lora_names_sample": [n for n, _ in S.lora[:3]], "projector_names": [n for n, _ in S.proj]},
            "student_trainable_sha256": sh.hexdigest(), "z_s_init_4_sum": float(z4.sum()), "z_s_init_4_sha256": hashlib.sha256(z4.numpy().tobytes()).hexdigest(),
            "student_frozen_numel": sum(p.numel() for p in S.peft.parameters() if not p.requires_grad), "teacher_trainable": sum(p.numel() for p in Tm.peft.parameters() if p.requires_grad),
            "args": vars(a)}
    assert meta["teacher_trainable"] == 0
    print("META", json.dumps(meta), flush=True)

    if a.mode == "audit":
        g = {"meta": meta}
        # --- divergence unit tests (random normalized distributions, fp32)
        torch.manual_seed(0); V = 1000; x = torch.randn(6, V, device=S.dev) * 3; yv = torch.randn(6, V, device=S.dev, requires_grad=True)
        lq, lp = F.log_softmax(x, -1), F.log_softmax(yv, -1)
        man = (lq.exp() * (lq - lp)).sum(-1); ref = F.kl_div(lp, lq, log_target=True, reduction="none").sum(-1)
        mm = 0.5 * (lq.exp() + lp.exp()); js_man = 0.5 * (lq.exp() * (lq - mm.log())).sum(-1) + 0.5 * (lp.exp() * (lp - mm.log())).sum(-1)
        gk = torch.autograd.grad(kl_fwd(x, yv).sum(), yv)[0]; gj = torch.autograd.grad(jsd(x.detach(), yv).sum(), yv)[0]
        g["unit"] = {"kl_self_zero": float(kl_fwd(x, x).abs().max()), "jsd_self_zero": float(jsd(x, x).abs().max()), "kl_nonneg": bool((kl_fwd(x, yv) >= -1e-6).all()),
                     "jsd_nonneg": bool((jsd(x, yv) >= -1e-6).all()), "kl_matches_F_kl_div_target_teacher": float((kl_fwd(x, yv) - ref).abs().max()),
                     "kl_matches_manual": float((kl_fwd(x, yv) - man).abs().max()), "kl_direction_not_reverse": float((kl_fwd(x, yv) - F.kl_div(lq, lp, log_target=True, reduction="none").sum(-1)).abs().max()),
                     "jsd_matches_manual": float((jsd(x, yv) - js_man).abs().max()), "kl_grad_finite_nonzero": bool(torch.isfinite(gk).all() and gk.abs().sum() > 0),
                     "jsd_grad_finite_nonzero": bool(torch.isfinite(gj).all() and gj.abs().sum() > 0), "jsd_le_ln2": bool((jsd(x, yv) <= math.log(2) + 1e-6).all())}
        # --- Gate 0A: teacher (Q + P_A / P_B, zero slots, new protocol) first-operation margins on the 7 valid held-out pairs
        rows0 = []
        with torch.no_grad():
            for xx in PS:
                if not xx["valid"]:
                    rows0.append({"id": xx["p"]["id"], "valid": False, "why": xx["why"]}); continue
                p, k = xx["p"], xx["k"]; s, dec = {}, {}
                z0 = torch.zeros(8, 2560, device=S.dev)
                for pn, plan in (("PA", p["plan_A_norm"]), ("PB", p["plan_B_norm"])):
                    for un, ids in (("A", xx["ia"]), ("B", xx["ib"])):
                        lg, y, e = fwd(Tm, tpre(Tm, p["question"], plan), z0, ids, len(ids)); assert y.tolist() == ids
                        s[(pn, un)] = span_lp(lg, ids, k)
                        if un == "A":
                            dec[pn] = lg[k]
                mA = s[("PA", "A")][0] - s[("PB", "A")][0]; mB = s[("PB", "B")][0] - s[("PA", "B")][0]
                rows0.append({"id": p["id"], "valid": True, "method_A": p["method_label_A"], "method_B": p["method_label_B"], "k": k, "c_tail": xx["c"][-80:],
                              "first_div_A": S.tok.decode(xx["ia"][k:k + 1]), "first_div_B": S.tok.decode(xx["ib"][k:k + 1]), "len_A": len(xx["ia"]) - k, "len_B": len(xx["ib"]) - k,
                              "margin_A": mA, "margin_B": mB, "both_pos": mA > 0 and mB > 0, "both_gt_0.01": mA > 0.01 and mB > 0.01,
                              "s_total": {f"{u}|{pn}": v[1] for (pn, u), v in s.items()}, "top1": {pn: S.tok.decode([int(d.argmax())]) for pn, d in dec.items()},
                              "jsd_PA_PB_decision": float(jsd(dec["PA"].unsqueeze(0), dec["PB"].unsqueeze(0))[0]), "U_A": xx["UA"], "U_B": xx["UB"]})
        V0 = [r for r in rows0 if r["valid"]]
        g["gate0A"] = {"rows": rows0, "n_valid": len(V0), "both_pos": sum(r["both_pos"] for r in V0), "both_gt_0.01": sum(r["both_gt_0.01"] for r in V0),
                       "mean_margin_A": float(np.mean([r["margin_A"] for r in V0])), "mean_margin_B": float(np.mean([r["margin_B"] for r in V0])),
                       "min_margin": float(min(min(r["margin_A"], r["margin_B"]) for r in V0)), "rule": ">= 5/7 with both margins > 0.01 nats/token"}
        g["gate0A"]["PASS"] = g["gate0A"]["both_gt_0.01"] >= 5
        # --- 4 fixed samples: terms, alignment, independent recompute, leak checks
        g["items"], g["leak"] = [], []
        with torch.no_grad():
            for s in B4:
                parts, info = item(S, Tm, s, n_tail); lt, lw = info["lt"], info["lw"]
                lq_, lp_ = F.log_softmax(lt.float(), -1), F.log_softmax(lw.float(), -1)
                ref_kd = (F.kl_div(lp_, lq_, log_target=True, reduction="none").sum(-1) if DIV == "forward_kl" else
                          0.5 * (lq_.exp() * (lq_ - (0.5 * (lq_.exp() + lp_.exp())).log())).sum(-1) + 0.5 * (lp_.exp() * (lp_ - (0.5 * (lq_.exp() + lp_.exp())).log())).sum(-1)).mean()
                sf, _, _ = S.score(s["question"], S.zlat(s["question"]).unsqueeze(0), [S.target_ids(s["execution"])])
                tp_txt, sp_txt = S.tok.decode(tpre(Tm, s["question"], s["plan_norm"])), S.tok.decode(student_pre(S, s["question"]))
                g["leak"].append({"id": s["id"], "teacher_has_plan": s["plan_norm"][:80] in tp_txt, "teacher_has_answer_hint": "Reference final answer" in tp_txt,
                                  "student_has_plan": s["plan_norm"][:80] in sp_txt or "Reference plan" in sp_txt, "student_has_answer_hint": "Reference final answer" in sp_txt})
                g["items"].append({"id": s["id"], **{k: float(v) for k, v in parts.items()}, "total": float(total(parts)), "nE": info["nE"], "aligned": info["aligned"],
                                   "ref_kd": float(ref_kd), "ref_exec_score": float(-sf[0]), "teacher_eval_mode": not Tm.peft.training, "student_eval_mode_no_dropout": not S.peft.training})
        g["grads"] = term_grads(S, Tm, B4, n_tail)
        # --- the KD gradient must reach the projector through Z_S: also check Z_S itself receives a gradient from KD alone
        s = B4[0]; q, tgt = s["question"], S.target_ids(s["execution"]); nE = len(tgt) - n_tail
        z = S.zlat(q); z.retain_grad(); lw, _, _ = fwd(S, student_pre(S, q), z, tgt, nE)
        with torch.no_grad():
            lt, _, _ = fwd(Tm, tpre(Tm, q, s["plan_norm"]), torch.zeros_like(z), tgt, nE)
        kd_fn(lt, lw).mean().backward(); g["kd_grad_on_ZS_norm"] = float(z.grad.norm()); g["ZS_requires_grad"] = bool(z.requires_grad)
        for p in params:
            p.grad = None
        g["decision0"], g["decision0_rows"] = decision_eval(S, Tm, PS)
        it, gr = g["items"], g["grads"]
        hard = {"unit_tests": g["unit"]["kl_self_zero"] < 1e-5 and g["unit"]["jsd_self_zero"] < 1e-5 and g["unit"]["kl_nonneg"] and g["unit"]["jsd_nonneg"]
                and g["unit"]["kl_matches_F_kl_div_target_teacher"] < 1e-4 and g["unit"]["jsd_matches_manual"] < 1e-4 and g["unit"]["kl_grad_finite_nonzero"] and g["unit"]["jsd_grad_finite_nonzero"]
                and g["unit"]["kl_direction_not_reverse"] > 1e-3,
                "gate0A": g["gate0A"]["PASS"], "finite": all(math.isfinite(x["total"]) for x in it), "aligned": all(x["aligned"] for x in it),
                "kd_matches_independent": max(abs(x["kd"] - x["ref_kd"]) for x in it) < 1e-4, "exec_matches_ref": max(abs(x["exec"] - x["ref_exec_score"]) for x in it) < 5e-3,
                "kd_to_projector_nonzero": gr["kd"]["projector"] > 0, "kd_to_lora_nonzero": gr["kd"]["lora"] > 0, "kd_grad_on_ZS": g["kd_grad_on_ZS_norm"] > 0,
                "anchor_exec_cos_to_projector": all(gr[k]["projector"] > 0 for k in ("anchor", "exec", "cos")), "all_grads_finite": all(gr[k]["finite"] for k in ("kd", "anchor", "exec", "cos", "total")),
                "teacher_and_frozen_grads_none": gr["teacher_grads_none"] and gr["frozen_student_grads_none"], "teacher_frozen_eval": meta["teacher_trainable"] == 0 and all(x["teacher_eval_mode"] for x in it),
                "no_dropout": all(x["student_eval_mode_no_dropout"] for x in it), "no_leak_to_student": not any(x["student_has_plan"] or x["student_has_answer_hint"] for x in g["leak"]),
                "teacher_plan_only": all(x["teacher_has_plan"] and not x["teacher_has_answer_hint"] for x in g["leak"])}
        g["hard"] = hard; g["ALL_HARD_PASS"] = all(hard.values())
        json.dump(g, open(f"{a.out}/audit_gate0" + ("" if a.lam_kd == 1.0 else f"_lamkd{a.lam_kd:g}") + ".json", "w"), indent=1, default=str); print(json.dumps({k: g[k] for k in ("unit", "hard", "ALL_HARD_PASS", "grads", "decision0")}, indent=1, default=str))
        print("GATE0A", json.dumps({k: v for k, v in g["gate0A"].items() if k != "rows"}))
        return

    json.dump(meta, open(f"{a.out}/run_meta.json", "w"), indent=1)
    allB = batches(singles, [], a.n_single, 0, a.steps, seed=a.data_seed)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    S_ = lambda x: {int(v) for v in x.split(",") if v}
    save_at, smoke_at, light_at, grad_at, interv_at, diag_at = map(S_, (a.save_at, a.smoke_at, a.light_at, a.grad_at, a.interv_at, a.diag_at))
    log = open(f"{a.out}/train.jsonl", "a"); t0 = time.time(); st0 = {}

    def checkpoint(step):
        save_ck(S, opt, f"{a.out}/checkpoints/stage2c_step{step}.pt", step, {"phase": "stage2c", "stage2c_local_step": step, "init": a.init, "distill_type": DIV,
                "init_global_step_2b": INIT_GLOBAL_STEP, "lambdas": LAMS, "lr": a.lr})

    def checks(step):
        tc = time.time(); smoke = step in smoke_at; final = step == a.steps
        S.peft.eval()
        res = {"step": step, "kind": "smoke16" if (smoke or final) else "light4", "distill_type": DIV}
        tfa, tfr, Z = tf_eval(S, Tm, ev, n_tail); res["tf16"] = tfa
        res["decision"], drows = decision_eval(S, Tm, PS)
        n_gen = len(ev) if (smoke or final) else 4
        with torch.no_grad():
            G = gen_rows(S, ev[:n_gen], Z[:n_gen], a.gen_tokens)
            res["gen"] = {"n": n_gen, "correct": sum(r["correct"] for r in G), "format_ok": sum(r["format_ok"] for r in G), "empty": sum(r["empty"] for r in G),
                          "degenerate": sum(r["degenerate"] for r in G), "first4_correct": sum(r["correct"] for r in G[:4]),
                          "first4_broken": sum(r["degenerate"] or r["empty"] or not r["format_ok"] for r in G[:4])}
            if step in interv_at:            # free-generation interventions on the same 16 questions: shuffled Z_S (seed-0 derangement) and zeros
                pm = derange(len(ev), 0); IV = {}
                for nm, ZZ in (("shuffle", [Z[pm[i]] for i in range(len(ev))]), ("zero", [torch.zeros_like(z) for z in Z])):
                    Gi = gen_rows(S, ev, ZZ, a.gen_tokens); Gw = G if n_gen == len(ev) else None
                    IV[nm] = {"correct": sum(r["correct"] for r in Gi), "format_ok": sum(r["format_ok"] for r in Gi), "degenerate": sum(r["degenerate"] for r in Gi),
                              "answer_change_k": sum(r["pred"] != w["pred"] for r, w in zip(Gi, Gw)), "identical_text_k": sum(r["text"] == w["text"] for r, w in zip(Gi, Gw)),
                              "prefix300_identical_k": sum(common_prefix(r["text"], w["text"]) >= 300 for r, w in zip(Gi, Gw)),
                              "common_prefix_chars": [common_prefix(r["text"], w["text"]) for r, w in zip(Gi, Gw)],
                              "with_correct_to_wrong": sum(w["correct"] and not r["correct"] for r, w in zip(Gi, Gw)), "with_wrong_to_correct": sum(r["correct"] and not w["correct"] for r, w in zip(Gi, Gw))}
                    with open(f"{a.out}/evaluations/freegen_{nm}_step{step}.jsonl", "w") as f:
                        for r in Gi:
                            f.write(json.dumps(r) + "\n")
                res["freegen_interv"] = IV
        if step in diag_at:
            d = diag(S, vals, 48, f"stage2c {DIV} local step {step}"); json.dump(d, open(f"{a.out}/evaluations/diag_stage2c_step{step}.json", "w"), indent=1)
            res["diag48"] = {"fr_paired": d["fr"]["paired"], "fr_gap_perm": d["fr"]["gap_perm"]["mean"], "fr_gap_ci": d["fr"]["gap_perm"]["ci95"],
                             "zs_crossq_cos": d["collapse"]["zs_crossq_cos"], "var_ratio": d["collapse"]["zs_var_over_teacher_var"]}
        if step in grad_at:
            res["grads4"] = term_grads(S, Tm, B4, n_tail)
        res["teacher_unchanged"] = teacher_unchanged()
        stop = None
        if step == 0:
            st0.update(res)
        else:
            b = st0; g0, g1 = b["gen"], res["gen"]; dn = res["tf16"]["eval_gold_nll"] - b["tf16"]["eval_gold_nll"]
            if smoke:
                fail = ((g0["correct"] - g1["correct"] >= 5 and g0["format_ok"] - g1["format_ok"] >= 6) or g1["first4_broken"] >= 3
                        or (dn >= 0.20 and g0["correct"] - g1["correct"] >= 4))
            else:
                fail = g1["first4_broken"] >= 3 or (dn >= 0.20 and g1["first4_correct"] <= g0["first4_correct"] - 2)
            res["gold_nll_increase"] = dn; res["safety_fail"] = bool(fail)
            if fail:
                stop = f"SAFETY_FAIL at step {step}"
        if not res["teacher_unchanged"]:
            stop = "INVALID_IMPLEMENTATION: teacher parameters changed"
        res["check_seconds"] = time.time() - tc
        json.dump(res, open(f"{a.out}/evaluations/check_step{step}.json", "w"), indent=1)
        with open(f"{a.out}/evaluations/check_rows_step{step}.jsonl", "w") as f:
            for r in tfr + [{"decision_row": 1, **r} for r in drows] + [{"gen_row": 1, **r} for r in G]:
                f.write(json.dumps(r) + "\n")
        print("CHECK", json.dumps({k: v for k, v in res.items() if k != "grads4"}), flush=True)
        return stop

    if 0 in save_at:
        checkpoint(0)
    if (stop := checks(0)):
        print("STOP:", stop); return
    for step in range(1, a.steps + 1):
        ts = time.time(); si, _ = allB[step - 1]; n = len(si)
        opt.zero_grad(set_to_none=True); rec = []; ntok = 0; aligned = True
        for i in si:
            parts, info = item(S, Tm, singles[i], n_tail)
            (total(parts) / n).backward(); rec.append({k: float(v) for k, v in parts.items()}); ntok += info["n_tgt"]; aligned &= info["aligned"]
        gp, gl = grad_norm([p for _, p in S.proj]), grad_norm([p for _, p in S.lora])
        fin = all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)
        t_nograd = all(p.grad is None for p in Tm.peft.parameters())
        before = [p.detach().clone() for p in params]
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0)); opt.step()
        npj = len(S.proj)
        dP = math.sqrt(sum(float((p.detach() - b).float().pow(2).sum()) for p, b in zip(params[:npj], before[:npj])))
        dL = math.sqrt(sum(float((p.detach() - b).float().pow(2).sum()) for p, b in zip(params[npj:], before[npj:]))); del before
        m = {k: float(np.mean([r[k] for r in rec])) for k in rec[0]}; dt = time.time() - ts
        tot = sum(LAMS[k] * v for k, v in m.items())
        r = {"step": step, "distill_type": DIV, "loss_kd_raw": m["kd"], "loss_anchor_signed": m["anchor"], "loss_exec_raw": m["exec"], "loss_cos_raw": m["cos"],
             **{f"loss_{k}_weighted": LAMS[k] * m[k] for k in LAMS}, "loss_total": tot,
             "grad_total_projector": gp, "grad_total_lora": gl, "grad_total_preclip": gn, "clipped": gn > 1.0, "delta_param_projector": dP, "delta_param_lora": dL,
             "lr": opt.param_groups[0]["lr"], "finite": fin, "labels_aligned": aligned, "teacher_grad_none": t_nograd, "train_tokens": ntok, "step_time_s": dt,
             "elapsed_s": time.time() - t0, "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30, "batch": si}
        log.write(json.dumps(r) + "\n"); log.flush()
        print(f"step {step} kd {m['kd']:.6f} anchor {m['anchor']:+.6f} exec {m['exec']:.4f} cos {m['cos']:.4f} total {tot:+.5f} gn {gn:.4f} (p {gp:.4f} l {gl:.4f}) "
              f"dW (p {dP:.2e} l {dL:.2e}) fin {fin} {dt:.1f}s {r['elapsed_s']:.0f}s", flush=True)
        if not fin or not math.isfinite(tot) or not aligned or not t_nograd or dP == 0.0 or dL == 0.0:
            print("STOP: INVALID_IMPLEMENTATION (non-finite / misaligned / teacher grad / no real update)"); checkpoint(f"{step}_stopped"); break
        if step in save_at:
            checkpoint(step)
        if (step in smoke_at or step in light_at or step == a.steps) and (stop := checks(step)):
            print("STOP:", stop); checkpoint(f"{step}_stopped") if step not in save_at else None; break


if __name__ == "__main__":
    main()
