"""Stage 2B latent-bypass repair: D0 (continued 2-loss) vs A (AnchorSim-inspired negative JSD) vs B (plan-aware contrastive).

All branches start from the SAME joint 2-loss step-50 checkpoint (projector + LoRA + AdamW state) and continue for the
same global steps 51..100 of the original single-sample order (batches(..., seed=7)); B additionally sees 2 verified
same-question A/B pairs per step (fixed order, seed=11).  Shared baseline (unchanged):
  L0 = 1.0 * L_cos^TF + 0.1 * L_exec^FR      (joint_2b: TF slot-i prediction from Q + Z_T[:i]; FR = free-running Z_S, no detach)
A: L0 + lam_anchor * L_anchor,  L_anchor = - mean_{t in T} JSD(p_aware,t || sg[p_blind,t])
   p_aware = p(. | Q, Z_S, E_<t) (the same forward as L_exec, gradient through LoRA + Z_S + projector)
   p_blind = p(. | Q, 0-vectors in the same 8 slots, E_<t)   (torch.no_grad; identical ids / positions / mask / targets)
   T = the first ANCHOR_T execution target tokens (early/middle CoT; never Q, slots, padding, '</EXECUTION><|im_end|>').
B: L0 + lam_plan * L_plan,  L_plan = L_matched + eta * L_rank   (frozen 2A teacher latents Z_T^A / Z_T^B, no plan text)
   s(E | Z) = mean log p over the strategy span of E (tokens after the token-LCP of E_A / E_B, up to '</EXECUTION>')
   D_A = s(E_A|Z_A) - s(E_A|Z_B), D_B = s(E_B|Z_B) - s(E_B|Z_A); L_rank = (softplus(-D_A/tau) + softplus(-D_B/tau)) / 2
   L_matched = (CE(E_A|Z_A) + CE(E_B|Z_B)) / 2 over the full target.  One forward with K = 4 rows; LoRA-only gradient.
Modes: audit (G0/G1 gate, no training) | train | probe | extra (JSD / same-Q 2x2 likelihood) | acc (batched greedy accuracy).
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from later.scripts.bridge.bridge_input_abcd import boot
from later.scripts.bridge.mini_bridge import correct, last_boxed
from later.scripts.stage2s.stochastic_pilot import L, build_pairs, grad_norm, key, short
from later.scripts.stage2t.joint_2b import LAM_COS, LAM_EXEC, cosd, tf_latents
from later.scripts.stage2t.three_step import D, TStudent, batches, load_jsonl, save_ck

ANCHOR_T = 128
N_TAIL = None          # number of '</EXECUTION><|im_end|>' tokens at the end of every target (set at start-up)
START_STEP = 50


# ---------------------------------------------------------------- shared forward
def exec_forward(S, q, Zs, targets, spans=None, n_logits=0):
    """Same layout / numerics as Student.score.  Zs [K,8,d].  -> s_full [K], s_span [K] or None,
    logits list (first n_logits target positions per row, fp32) and aux (ids / pos / att / emb slots for the G0 checks)."""
    pre = S.prompt_ids(q)
    slots = list(range(len(pre), len(pre) + L))
    pre = pre + [S.T.pad_id] * L + S.T.suffix_ids
    K = len(targets); n = len(pre) + max(len(t) for t in targets)
    ids = torch.full((K, n), S.T.pad_id, dtype=torch.long); att = torch.zeros(K, n, dtype=torch.long)
    for i, t in enumerate(targets):
        ids[i, :len(pre) + len(t)] = torch.tensor(pre + t); att[i, :len(pre) + len(t)] = 1
    ids, att = ids.to(S.dev), att.to(S.dev)
    emb = S.base.get_input_embeddings()(ids)
    emb = torch.cat([emb[:, :slots[0]], Zs.to(emb.dtype), emb[:, slots[-1] + 1:]], 1)
    pos = torch.arange(n, device=S.dev).unsqueeze(0).expand(K, -1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hs = S.base.model(inputs_embeds=emb, attention_mask=att, position_ids=pos, use_cache=False).last_hidden_state
    s_full, s_span, lgs = [], [], []
    for i, t in enumerate(targets):
        lg = S.base.lm_head(hs[i, len(pre) - 1:len(pre) + len(t) - 1]).float()
        lp = -F.cross_entropy(lg, torch.tensor(t, device=S.dev), reduction="none")
        s_full.append(lp.mean())
        if spans is not None:
            a, b = spans[i]; s_span.append(lp[a:b].mean())
        if n_logits:
            lgs.append(lg[:n_logits])
    return torch.stack(s_full), (torch.stack(s_span) if spans is not None else None), lgs, \
        {"ids": ids, "pos": pos, "att": att, "emb": emb, "slots": slots, "n_pre": len(pre)}


def token_jsd(logits_aware, logits_blind):
    """[T,V] x [T,V] -> [T] JSD in nats (fp32); the blind distribution carries no gradient."""
    p = torch.softmax(logits_aware.float(), -1)
    with torch.no_grad():
        q = torch.softmax(logits_blind.float(), -1)
    m = (p + q) / 2
    log_m = m.clamp_min(1e-12).log()
    js = 0.5 * (p * (p.clamp_min(1e-12).log() - log_m)).sum(-1)
    js = js + 0.5 * (q * (q.clamp_min(1e-12).log() - log_m)).sum(-1)
    return js


def n_anchor(tgt):
    return max(1, min(ANCHOR_T, len(tgt) - N_TAIL))


def blind_latent(Z):
    return torch.zeros_like(Z)                     # content mask: same 8 slots / positions, zero vectors


# ---------------------------------------------------------------- losses
def single_loss(S, s, lam_anchor):
    q, T = s["question"], S.teacher(s["plan_norm"])
    l_cos = cosd(tf_latents(S, q, T), T)
    Z = S.zlat(q); tgt = S.target_ids(s["execution"])
    nT = n_anchor(tgt) if lam_anchor > 0 else 0
    sf, _, lg, _ = exec_forward(S, q, Z.unsqueeze(0), [tgt], n_logits=nT)
    l_exec = -sf[0]
    parts = {"cos": l_cos, "exec": l_exec}
    loss = LAM_COS * l_cos + LAM_EXEC * l_exec
    if lam_anchor > 0:
        with torch.no_grad():
            _, _, lgb, _ = exec_forward(S, q, blind_latent(Z).unsqueeze(0), [tgt], n_logits=nT)
        jsd = token_jsd(lg[0], lgb[0])
        parts["anchor"] = -jsd.mean()
        loss = loss + lam_anchor * parts["anchor"]
    with torch.no_grad():
        info = {"fr_cosd": float(cosd(Z, T)), "adj_cos": float(F.cosine_similarity(Z[1:], Z[:-1], -1).mean())}
    return loss, parts, info


def strategy_span(p, X):
    tgt = p["tgt_" + X]; k = len(p["C"])
    return (k, max(k + 1, len(tgt) - N_TAIL))


def plan_terms(S, p, tau, eta):
    TA, TB = S.teacher(p["plan_A_norm"]), S.teacher(p["plan_B_norm"])
    tA, tB = p["tgt_A"], p["tgt_B"]
    spA, spB = strategy_span(p, "A"), strategy_span(p, "B")
    sf, ss, _, _ = exec_forward(S, p["question"], torch.stack([TA, TB, TB, TA]), [tA, tA, tB, tB], [spA, spA, spB, spB])
    dA, dB = ss[0] - ss[1], ss[2] - ss[3]
    l_rank = 0.5 * (F.softplus(-dA / tau) + F.softplus(-dB / tau))
    l_matched = -0.5 * (sf[0] + sf[2])
    return l_matched + eta * l_rank, {"matched": l_matched, "rank": l_rank, "dA": dA, "dB": dB}


# ---------------------------------------------------------------- data
def train_pairs(S, s2b):
    pairs = build_pairs(S, s2b)
    return [p for p in pairs if p["role"] == "train_pair" and not p["excluded"]], [p for p in pairs if p["role"] == "heldout8"], pairs


def pair_order(n_pairs, n_per_step, steps, seed=11):
    rng = random.Random(seed); po, out = [], []
    for _ in range(steps):
        while len(po) < n_per_step:
            i = list(range(n_pairs)); rng.shuffle(i); po += i
        out.append([po.pop(0) for _ in range(n_per_step)])
    return out


def norm_q(s):
    return " ".join(s.lower().split())


def acc_questions(s2b, n, seed=0):
    """n unseen questions from stage2_val.parquet, excluding every question used in training or in the 48-question set."""
    used = set()
    for f in ("train_singles", "val_singles", "pairs"):
        for r in load_jsonl(f"{s2b}/data/{f}.jsonl"):
            used.add(norm_q(r["question"]))
    df = pd.read_parquet(os.path.join(os.path.dirname(os.path.dirname(s2b)), "data", "stage2_val.parquet"))
    df = df.drop_duplicates("question")
    cand = [r for r in df.to_dict("records") if norm_q(r["question"]) not in used]
    rng = random.Random(seed); rng.shuffle(cand)
    return [{"record_id": r["record_id"], "question": r["question"], "answer": str(r["answer"])} for r in cand[:n]], len(cand)


# ---------------------------------------------------------------- batched greedy generation over questions
@torch.no_grad()
def generate_batch(S, qs, Zs, max_new=2048):
    pres = [S.prompt_ids(q) for q in qs]
    full = [p + [S.T.pad_id] * L + S.T.suffix_ids for p in pres]
    n = max(len(f) for f in full); K = len(qs)
    ids = torch.full((K, n), S.T.pad_id, dtype=torch.long); att = torch.zeros(K, n, dtype=torch.long)
    for i, f in enumerate(full):
        ids[i, n - len(f):] = torch.tensor(f); att[i, n - len(f):] = 1
    ids, att = ids.to(S.dev), att.to(S.dev)
    emb = S.base.get_input_embeddings()(ids).clone()
    for i, p in enumerate(pres):
        o = n - len(full[i]) + len(p); emb[i, o:o + L] = Zs[i].to(emb.dtype)
    pos = (att.cumsum(1) - 1).clamp_min(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = S.base.model(inputs_embeds=emb, attention_mask=att, position_ids=pos, use_cache=True)
        cache = out.past_key_values; logits = S.base.lm_head(out.last_hidden_state[:, -1]).float()
        eos = torch.tensor(S.eos, device=S.dev); done = torch.zeros(K, dtype=torch.bool, device=S.dev)
        gen = [[] for _ in range(K)]; last = pos[:, -1:]
        for t in range(max_new):
            tk = logits.argmax(-1)
            for i in range(K):
                if not done[i]:
                    gen[i].append(int(tk[i]))
            done = done | torch.isin(tk, eos)
            if bool(done.all()):
                break
            att = torch.cat([att, torch.ones(K, 1, dtype=att.dtype, device=S.dev)], 1); last = last + 1
            out = S.base.model(input_ids=tk.unsqueeze(1), attention_mask=att, position_ids=last, past_key_values=cache, use_cache=True)
            cache = out.past_key_values; logits = S.base.lm_head(out.last_hidden_state[:, -1]).float()
    return [S.tok.decode(g, skip_special_tokens=False) for g in gen]


def st(v):
    v = [float(x) for x in v]
    return {"mean": float(np.mean(v)), "ci95": boot(v), "frac_pos": float(np.mean([x > 0 for x in v])), "n": len(v)}


# ---------------------------------------------------------------- main
def main():
    global N_TAIL
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["audit", "train", "probe", "extra", "acc"], required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--s2b", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None, help="joint 2-loss step-50 checkpoint (projector + LoRA + AdamW)")
    ap.add_argument("--resume", default=None, help="own checkpoint to continue (same branch)")
    ap.add_argument("--tag", default="D0"); ap.add_argument("--lam_anchor", type=float, default=0.0); ap.add_argument("--lam_plan", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=1.0); ap.add_argument("--eta", type=float, default=1.0); ap.add_argument("--n_pair", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50); ap.add_argument("--stop_at", type=int, default=50); ap.add_argument("--save_at", default="10,25,50")
    ap.add_argument("--n_single", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ckpt", default=None); ap.add_argument("--n_val", type=int, default=48); ap.add_argument("--n_acc", type=int, default=128)
    ap.add_argument("--acc_bs", type=int, default=32); ap.add_argument("--gen_tokens", type=int, default=2048)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = TStudent(a.model, a.s2b, seed=0)
    N_TAIL = len(S.tok.encode("</EXECUTION><|im_end|>", add_special_tokens=False))
    singles = load_jsonl(f"{a.s2b}/data/train_singles.jsonl"); vals = load_jsonl(f"{a.s2b}/data/val_singles.jsonl")
    trainp, held, allp = train_pairs(S, a.s2b)
    params = S.params()

    if a.mode == "audit":
        S.load(torch.load(a.ckpt, weights_only=False)["state"]); g = {"ckpt": a.ckpt}
        ck = torch.load(a.ckpt, weights_only=False)
        g["ckpt_step"] = ck.get("step"); g["ckpt_has_opt"] = ck.get("opt") is not None; g["ckpt_meta"] = {k: ck.get(k) for k in ("phase", "lam_cos", "lam_exec", "lr")}
        g["trainable"] = {"projector": sum(p.numel() for _, p in S.proj), "lora": sum(p.numel() for _, p in S.lora)}
        # ---- G0: representation stats (48 val)
        V = vals[:a.n_val]
        with torch.no_grad():
            ZT = torch.stack([S.teacher(v["plan_norm"]) for v in V]); ZS = torch.stack([S.zlat(v["question"]) for v in V])
        g["G0_shapes"] = [list(ZT.shape), list(ZS.shape)]
        for nm, Z in (("teacher", ZT), ("student", ZS)):
            g[f"G0_{nm}_slot_norm"] = [float(x) for x in Z.norm(dim=-1).mean(0)]
            g[f"G0_{nm}_slot_rms"] = [float(x) for x in Z.pow(2).mean(-1).sqrt().mean(0)]
            g[f"G0_{nm}_slot_mean_abs"] = [float(x) for x in Z.mean(0).abs().mean(-1)]
            g[f"G0_{nm}_slot_var"] = [float(x) for x in Z.var(0).mean(-1)]
        g["G0_tok_rms"] = S.tok_rms
        # ---- G0: aware / blind layout identical except slot content; latent actually written into the embeddings
        s0 = singles[0]; q = s0["question"]; tgt = S.target_ids(s0["execution"]); nT = n_anchor(tgt)
        with torch.no_grad():
            Z = S.zlat(q)
            sa, _, la, ax = exec_forward(S, q, Z[None], [tgt], n_logits=nT)
            sb, _, lb, bx = exec_forward(S, q, blind_latent(Z)[None], [tgt], n_logits=nT)
            sref, _, _ = S.score(q, Z[None], [tgt])
        sl = ax["slots"]
        nonslot = torch.ones(ax["emb"].size(1), dtype=torch.bool); nonslot[sl] = False
        g["G0_same_ids_pos_att"] = bool(torch.equal(ax["ids"], bx["ids"]) and torch.equal(ax["pos"], bx["pos"]) and torch.equal(ax["att"], bx["att"]))
        g["G0_nonslot_embeddings_identical"] = bool(torch.equal(ax["emb"][:, nonslot], bx["emb"][:, nonslot]))
        g["G0_aware_slots_hold_ZS"] = bool(torch.equal(ax["emb"][0, sl], Z.to(ax["emb"].dtype)))
        g["G0_blind_slots_zero"] = bool((bx["emb"][0, sl] == 0).all())
        g["G0_exec_forward_equals_Student_score"] = [float(sa[0]), float(sref[0])]
        g["G0_anchor_T"] = nT; g["G0_n_tail"] = N_TAIL; g["G0_T_tokens_decoded_tail"] = S.tok.decode(tgt[nT - 5:nT])
        g["G0_target_tail_excluded"] = S.tok.decode(tgt[-N_TAIL:])
        g["G0_jsd_aware_blind_mean"] = float(token_jsd(la[0], lb[0]).mean())
        g["G0_ce_aware_blind"] = [-float(sa[0]), -float(sb[0])]
        # ---- G1: same-question A/B teacher latents
        rows = []
        cs = lambda X, Y: float(F.cosine_similarity(X.flatten(), Y.flatten(), 0))
        for p in allp:
            TA, TB = S.teacher(p["plan_A_norm"]), S.teacher(p["plan_B_norm"])
            rows.append({"id": short(p["id"]), "role": p["role"], "excluded": p["excluded"], "choice_valid": p["choice_valid"],
                         "same_question": True, "cos_TA_TB": cs(TA, TB), "slot_cos_TA_TB": [float(x) for x in F.cosine_similarity(TA, TB, -1)],
                         "plan_hash_differs": key(p["plan_A_norm"]) != key(p["plan_B_norm"]),
                         "lcp_tokens": len(p["C"]), "len_A": len(p["tgt_A"]), "len_B": len(p["tgt_B"]),
                         "strategy_span_A": strategy_span(p, "A"), "strategy_span_B": strategy_span(p, "B"),
                         "method_label_A": p.get("method_label_A"), "method_label_B": p.get("method_label_B")})
        cross = [cs(S.teacher(allp[i]["plan_A_norm"]), S.teacher(allp[(i + 1) % len(allp)]["plan_A_norm"])) for i in range(len(allp))]
        g["G1_pairs"] = rows
        g["G1_sameQ_cos_TA_TB"] = {"mean": float(np.mean([r["cos_TA_TB"] for r in rows])), "min": float(np.min([r["cos_TA_TB"] for r in rows])),
                                   "max": float(np.max([r["cos_TA_TB"] for r in rows]))}
        g["G1_crossQ_cos"] = {"mean": float(np.mean(cross)), "min": float(np.min(cross)), "max": float(np.max(cross))}
        g["G1_n"] = {"train_pairs_used": len(trainp), "heldout": len(held), "excluded": [short(p["id"]) for p in allp if p["excluded"]],
                     "train_choice_valid": sum(p["choice_valid"] for p in trainp), "heldout_choice_valid": sum(p["choice_valid"] for p in held)}
        tq = {norm_q(p["question"]) for p in trainp}; hq = {norm_q(p["question"]) for p in held}
        sq = {norm_q(s["question"]) for s in singles}; vq = {norm_q(v["question"]) for v in vals}
        g["G1_split_overlaps"] = {"trainpair_heldout": len(tq & hq), "trainpair_singles": len(tq & sq), "heldout_singles": len(hq & sq),
                                  "heldout_val": len(hq & vq), "trainpair_val": len(tq & vq)}
        AQ, n_cand = acc_questions(a.s2b, a.n_acc)
        g["G2_acc_questions"] = {"n": len(AQ), "candidates_after_dedup": n_cand, "source": "stage2_val.parquet (seed 0 shuffle), excluding train_singles/val_singles/pairs questions"}
        with open(f"{a.out}/acc_questions.jsonl", "w") as f:
            for r in AQ:
                f.write(json.dumps(r) + "\n")
        json.dump(g, open(f"{a.out}/audit_gate.json", "w"), indent=1); print(json.dumps({k: v for k, v in g.items() if k != "G1_pairs"}, indent=1))
        return

    if a.mode == "train":
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        src = a.resume or a.init
        ck = torch.load(src, weights_only=False); S.load(ck["state"]); opt.load_state_dict(ck["opt"])
        done = ck["step"] - START_STEP if a.resume else 0          # additional steps already taken on this branch
        assert a.resume or ck["step"] == START_STEP
        allB = batches(singles, [], a.n_single, 0, START_STEP + a.steps, seed=7)[START_STEP:]   # global steps 51..100
        PO = pair_order(len(trainp), a.n_pair, a.steps)
        save_at = {int(x) for x in a.save_at.split(",")}
        os.makedirs(f"{a.out}/checkpoints", exist_ok=True)
        log = open(f"{a.out}/train_{a.tag}.jsonl", "a"); t0 = time.time(); torch.cuda.reset_peak_memory_stats(); ntok = 0
        use_plan = a.lam_plan > 0
        for step in range(done + 1, a.stop_at + 1):
            si = allB[step - 1][0]; pi = PO[step - 1] if use_plan else []
            gdiag = None
            if step in (1, 10, 25, 50):            # per-term weighted gradient norms (item 0 / pair 0, separate backward)
                gdiag = {}
                terms = [("cos", LAM_COS), ("exec", LAM_EXEC)] + ([("anchor", a.lam_anchor)] if a.lam_anchor > 0 else [])
                for nm, w in terms:
                    opt.zero_grad(set_to_none=True); _, parts, _ = single_loss(S, singles[si[0]], a.lam_anchor); (w * parts[nm]).backward()
                    gdiag[nm + "_weighted"] = {"projector": grad_norm([p for _, p in S.proj]), "lora": grad_norm([p for _, p in S.lora]), "raw": float(parts[nm])}
                if use_plan:
                    for nm in ("matched", "rank"):
                        opt.zero_grad(set_to_none=True); _, pp = plan_terms(S, trainp[pi[0]], a.tau, a.eta)
                        w = a.lam_plan * (a.eta if nm == "rank" else 1.0); (w * pp[nm]).backward()
                        gdiag["plan_" + nm + "_weighted"] = {"projector": grad_norm([p for _, p in S.proj]), "lora": grad_norm([p for _, p in S.lora]), "raw": float(pp[nm])}
            opt.zero_grad(set_to_none=True); recs, precs = [], []
            for i in si:
                loss, parts, info = single_loss(S, singles[i], a.lam_anchor); (loss / len(si)).backward()
                recs.append({k: float(v) for k, v in parts.items()} | info); ntok += len(S.target_ids(singles[i]["execution"]))
            for j in pi:
                lp, pp = plan_terms(S, trainp[j], a.tau, a.eta); (a.lam_plan * lp / len(pi)).backward()
                precs.append({k: float(v) for k, v in pp.items()} | {"id": short(trainp[j]["id"])}); ntok += len(trainp[j]["tgt_A"]) + len(trainp[j]["tgt_B"])
            gp, gl = grad_norm([p for _, p in S.proj]), grad_norm([p for _, p in S.lora])
            fin = all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0)); opt.step()
            m = lambda R, k: float(np.mean([r[k] for r in R])) if R and k in R[0] else None
            rec = {"step_additional": step, "step_global": START_STEP + step, "tag": a.tag, "lam_anchor": a.lam_anchor, "lam_plan": a.lam_plan,
                   "loss_cos_raw": m(recs, "cos"), "loss_exec_raw": m(recs, "exec"), "loss_anchor_raw": m(recs, "anchor"),
                   "jsd_aware_blind": -m(recs, "anchor") if a.lam_anchor > 0 else None,
                   "loss_plan_matched_raw": m(precs, "matched"), "loss_plan_rank_raw": m(precs, "rank"), "plan_dA": m(precs, "dA"), "plan_dB": m(precs, "dB"),
                   "loss_cos_weighted": LAM_COS * m(recs, "cos"), "loss_exec_weighted": LAM_EXEC * m(recs, "exec"),
                   "loss_anchor_weighted": a.lam_anchor * m(recs, "anchor") if a.lam_anchor > 0 else 0.0,
                   "loss_plan_weighted": a.lam_plan * (m(precs, "matched") + a.eta * m(precs, "rank")) if use_plan else 0.0,
                   "fr_cosd": m(recs, "fr_cosd"), "adj_cos": m(recs, "adj_cos"), "grad_total_preclip": gn, "grad_proj": gp, "grad_lora": gl, "grads_finite": fin,
                   "grad_per_term_item0": gdiag, "elapsed_s": time.time() - t0, "train_tokens_cum_this_session": ntok,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "batch": si, "pairs": [short(trainp[j]["id"]) for j in pi]}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"[{a.tag}] +{step} cos {rec['loss_cos_raw']:.4f} fr {rec['fr_cosd']:.4f} exec {rec['loss_exec_raw']:.4f} jsd {rec['jsd_aware_blind']} "
                  f"plan m {rec['loss_plan_matched_raw']} r {rec['loss_plan_rank_raw']} dA {rec['plan_dA']} dB {rec['plan_dB']} "
                  f"gn {gn:.3f} (p {gp:.4f} l {gl:.4f}) {gdiag or ''} {rec['elapsed_s']:.0f}s mem {rec['peak_mem_gb']:.0f}G", flush=True)
            if not fin or not np.isfinite(rec["loss_exec_raw"]):
                print("STOP: non-finite"); break
            if step in save_at or step == a.stop_at:
                save_ck(S, opt, f"{a.out}/checkpoints/{a.tag}_add{step}.pt", START_STEP + step,
                        {"tag": a.tag, "init": a.init, "lam_cos": LAM_COS, "lam_exec": LAM_EXEC, "lam_anchor": a.lam_anchor, "lam_plan": a.lam_plan,
                         "tau": a.tau, "eta": a.eta, "n_pair": a.n_pair, "lr": a.lr, "anchor_T": ANCHOR_T, "blind": "zero vectors in the 8 slots"})
        return

    S.load(torch.load(a.ckpt, weights_only=False)["state"]); name = os.path.basename(a.ckpt).replace(".pt", "")
    os.makedirs(f"{a.out}/evaluations", exist_ok=True)

    if a.mode == "probe":          # TF / FR cosine on the fixed 48 unseen questions (per-sample + per-slot kept)
        V = vals[:a.n_val]
        with torch.no_grad():
            T = torch.stack([S.teacher(v["plan_norm"]) for v in V])
            Ptf = torch.stack([tf_latents(S, v["question"], T[i]) for i, v in enumerate(V)]); Zs = torch.stack([S.zlat(v["question"]) for v in V])
        sh = torch.roll(torch.arange(len(V)), 1)
        c8 = lambda X, Y: F.cosine_similarity(X, Y, dim=-1)
        tf_p, tf_s, fr_p, fr_s = c8(Ptf, T), c8(Ptf, T[sh]), c8(Zs, T), c8(Zs, T[sh])
        fl = Zs.flatten(1); cc = F.cosine_similarity(fl.unsqueeze(1), fl.unsqueeze(0), -1); iu = torch.triu_indices(len(V), len(V), 1)
        res = {"ckpt": a.ckpt, "n": len(V), "tf_cos_paired": float(tf_p.mean()), "tf_cos_shuffled": float(tf_s.mean()), "tf_gap": st((tf_p - tf_s).mean(-1).tolist()),
               "fr_cos_paired": float(fr_p.mean()), "fr_cos_shuffled": float(fr_s.mean()), "fr_gap": st((fr_p - fr_s).mean(-1).tolist()),
               "teacher_mean_cos_baseline": float(c8(T.mean(0, keepdim=True).expand_as(T), T).mean()),
               "tf_gap_per_slot": [float(x) for x in (tf_p - tf_s).mean(0)], "fr_gap_per_slot": [float(x) for x in (fr_p - fr_s).mean(0)],
               "tf_cos_per_slot": [float(x) for x in tf_p.mean(0)], "fr_cos_per_slot": [float(x) for x in fr_p.mean(0)],
               "zs_crossq_cos": float(cc[iu[0], iu[1]].mean()), "zs_var_over_teacher_var": float(Zs.var(0).mean() / T.var(0).mean()),
               "zs_var_over_teacher_var_per_slot": [float(x) for x in Zs.var(0).mean(-1) / T.var(0).mean(-1)],
               "per_sample": {"tf_paired": tf_p.mean(-1).tolist(), "tf_shuffled": tf_s.mean(-1).tolist(), "fr_paired": fr_p.mean(-1).tolist(), "fr_shuffled": fr_s.mean(-1).tolist()}}
        json.dump(res, open(f"{a.out}/evaluations/probe_{name}.json", "w"), indent=1)
        print(json.dumps({k: v for k, v in res.items() if k != "per_sample"}, indent=1))
        return

    if a.mode == "extra":         # aware/blind/cross JSD + gold-CE interventions (48 val) ; same-Q A/B 2x2 likelihood (27 pairs)
        V = vals[:a.n_val]; vr = []
        with torch.no_grad():
            ZS = torch.stack([S.zlat(v["question"]) for v in V])
            for i, v in enumerate(V):
                q = v["question"]; tgt = S.target_ids(v["execution"]); nE = len(tgt) - N_TAIL; nT = n_anchor(tgt)
                conds = {"self": ZS[i], "zero": torch.zeros_like(ZS[i]), "crossq": ZS[(i + 1) % len(V)]}
                sf, _, lg, _ = exec_forward(S, q, torch.stack(list(conds.values())), [tgt] * 3, n_logits=nE)
                r = {"id": v["id"], "ce": dict(zip(conds, (-sf).tolist()))}
                for c in ("zero", "crossq"):
                    js = token_jsd(lg[0], lg[list(conds).index(c)])
                    r[f"jsd_self_{c}_T"] = float(js[:nT].mean()); r[f"jsd_self_{c}_fullE"] = float(js.mean())
                vr.append(r)
        out = {"ckpt": a.ckpt, "val48": {"jsd_aware_blind_T": st([r["jsd_self_zero_T"] for r in vr]), "jsd_aware_blind_fullE": st([r["jsd_self_zero_fullE"] for r in vr]),
                                         "jsd_self_crossq_T": st([r["jsd_self_crossq_T"] for r in vr]), "jsd_self_crossq_fullE": st([r["jsd_self_crossq_fullE"] for r in vr]),
                                         "gold_ce_self": float(np.mean([r["ce"]["self"] for r in vr])),
                                         "dCE_zero": st([r["ce"]["zero"] - r["ce"]["self"] for r in vr]), "dCE_crossq": st([r["ce"]["crossq"] - r["ce"]["self"] for r in vr])}}
        pr = []
        with torch.no_grad():
            for p in allp:
                TA, TB = S.teacher(p["plan_A_norm"]), S.teacher(p["plan_B_norm"]); Zs = S.zlat(p["question"])
                spA, spB = strategy_span(p, "A"), strategy_span(p, "B")
                C, DA, DB = p["C"], p["D_A"], p["D_B"]
                Zr = torch.stack([TA, TB, TA, TB, Zs, Zs])
                sf, ss, _, _ = exec_forward(S, p["question"], Zr, [p["tgt_A"], p["tgt_A"], p["tgt_B"], p["tgt_B"], p["tgt_A"], p["tgt_B"]], [spA, spA, spB, spB, spA, spB])
                _, sd, _, _ = exec_forward(S, p["question"], Zr[:4], [C + DA, C + DA, C + DB, C + DB], [(len(C), len(C) + len(DA))] * 2 + [(len(C), len(C) + len(DB))] * 2)
                g = lambda x: [float(y) for y in x]
                r = {"id": short(p["id"]), "role": p["role"], "excluded": p["excluded"], "choice_valid": p["choice_valid"],
                     "full": dict(zip(["EA_ZA", "EA_ZB", "EB_ZA", "EB_ZB", "EA_ZS", "EB_ZS"], g(sf))),
                     "span": dict(zip(["EA_ZA", "EA_ZB", "EB_ZA", "EB_ZB", "EA_ZS", "EB_ZS"], g(ss))),
                     "dspan": dict(zip(["EA_ZA", "EA_ZB", "EB_ZA", "EB_ZB"], g(sd)))}
                for kk in ("full", "span", "dspan"):
                    x = r[kk]; r[f"{kk}_gap"] = ((x["EA_ZA"] - x["EA_ZB"]) + (x["EB_ZB"] - x["EB_ZA"])) / 2
                    r[f"{kk}_both_correct"] = x["EA_ZA"] > x["EA_ZB"] and x["EB_ZB"] > x["EB_ZA"]
                pr.append(r)
        for role in ("train_pair", "heldout8"):
            R = [r for r in pr if r["role"] == role and not r["excluded"]]
            o = {"n": len(R)}
            for kk in ("full", "span", "dspan"):
                RR = R if kk != "dspan" else [r for r in R if r["choice_valid"]]
                o[f"{kk}_matched_minus_swapped"] = st([r[f"{kk}_gap"] for r in RR]); o[f"{kk}_both_correct"] = int(sum(r[f"{kk}_both_correct"] for r in RR))
                o[f"{kk}_n"] = len(RR)
                o[f"{kk}_2x2_mean"] = {c: float(np.mean([r[kk][c] for r in RR])) for c in ("EA_ZA", "EA_ZB", "EB_ZA", "EB_ZB")}
            out[role] = o
        json.dump(out, open(f"{a.out}/evaluations/extra_{name}.json", "w"), indent=1)
        with open(f"{a.out}/evaluations/extra_rows_{name}.jsonl", "w") as f:
            for r in vr:
                f.write(json.dumps({"kind": "val48", **r}) + "\n")
            for r in pr:
                f.write(json.dumps({"kind": "pair", **r}) + "\n")
        print(json.dumps(out, indent=1))
        return

    if a.mode == "acc":           # batched greedy accuracy on independent unseen questions (self latents)
        AQ = load_jsonl(f"{a.out}/acc_questions.jsonl")[:a.n_acc]; rows = []; t0 = time.time()
        for b0 in range(0, len(AQ), a.acc_bs):
            Bq = AQ[b0:b0 + a.acc_bs]
            with torch.no_grad():
                Zs = torch.stack([S.zlat(r["question"]) for r in Bq])
            gens = generate_batch(S, [r["question"] for r in Bq], Zs, a.gen_tokens)
            for r, g_ in zip(Bq, gens):
                pred = last_boxed(g_)
                rows.append({"record_id": r["record_id"], "answer": r["answer"], "pred": pred, "correct": correct(pred, r["answer"]),
                             "ended": "<|im_end|>" in g_, "exec_close": "</EXECUTION>" in g_, "n_chars": len(g_), "text": g_})
            print(f"{b0 + len(Bq)}/{len(AQ)} acc {np.mean([r['correct'] for r in rows]):.3f} {time.time() - t0:.0f}s", flush=True)
        c = [int(r["correct"]) for r in rows]
        res = {"ckpt": a.ckpt, "n": len(rows), "accuracy": float(np.mean(c)), "n_correct": int(sum(c)), "ci95": boot(c),
               "format_ok": int(sum(r["ended"] and r["exec_close"] for r in rows)), "no_boxed": int(sum(r["pred"] is None for r in rows)),
               "decoding": f"greedy, max {a.gen_tokens} new tokens, batch {a.acc_bs} (left padding), self latents", "wall_s": time.time() - t0}
        json.dump(res, open(f"{a.out}/evaluations/acc_{name}.json", "w"), indent=1)
        with open(f"{a.out}/evaluations/acc_rows_{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
