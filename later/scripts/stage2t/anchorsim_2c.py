"""Stage 2C AnchorSim two-loss smoke test (fixed answer-conditioned teacher, no EMA, <= 10 steps).

Student = joint 2-loss step-50 checkpoint (LoRA + projector trainable, Qwen3-4B base frozen).
Teacher = a second, fully frozen copy of the SAME checkpoint; it additionally sees the gold answer a* in its user turn.
For every valid execution token t (E text only: no prompt / slots / padding / '</EXECUTION><|im_end|>'):
  p_with  = pi_S(. | Q, z_S, E_<t)                   (z_S = free-running student latents, graph kept)
  p_blind = pi_S(. | Q, 0-vectors in the 8 slots, E_<t)   (no_grad; identical ids / positions / labels)
  p_T     = pi_T(. | Q + a*, sg(z_S), E_<t)          (no_grad; aligned on the execution-token index, not raw index)
  L = 1.0 * mean_t JSD(sg p_T, p_with) - 1.0 * mean_t JSD(p_with, sg p_blind)
Modes: audit (0-step, no optimizer step) | train (<= 10 steps, eval at 0/5/10).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from later.scripts.bridge.mini_bridge import correct, last_boxed
from later.scripts.stage2s.stochastic_pilot import L, grad_norm
from later.scripts.stage2t.contrastive_2b import generate_batch
from later.scripts.stage2t.three_step import TStudent, load_jsonl, save_ck
from later.src.train.stage2_plan_dataset import render_prompt

LAM_DISTILL, LAM_ANCHOR = 1.0, 1.0
HINT = "\n\n(Reference final answer, visible to the teacher only: ${}$)"


def jsd(la, lb):
    """[T,V] logits x [T,V] logits -> [T] JSD (nats), fp32 log_softmax; gradients flow through whichever input is not detached."""
    lp, lq = F.log_softmax(la.float(), -1), F.log_softmax(lb.float(), -1)
    lm = torch.logaddexp(lp, lq) - math.log(2.0)
    return 0.5 * (lp.exp() * (lp - lm)).sum(-1) + 0.5 * (lq.exp() * (lq - lm)).sum(-1)


def student_pre(S, q):
    return S.prompt_ids(q)


def teacher_pre(S, q, ans):
    return S.tok.encode(render_prompt(S.tok, q + HINT.format(ans)), add_special_tokens=False) + S.T.plan_open_ids


def fwd(M, pre, Z, tgt, nE, slots=True):
    """One sequence: pre + [8 slots] + suffix + tgt (slots=False: no slot positions = 'drop').
    -> logits for the first nE target tokens [nE,V] (row t predicts tgt[t]) and the label ids actually sitting at those positions."""
    full = pre + ([M.T.pad_id] * L if slots else []) + M.T.suffix_ids
    ids = torch.tensor([full + tgt], device=M.dev)
    emb = M.base.get_input_embeddings()(ids)
    if slots:
        emb = torch.cat([emb[:, :len(pre)], Z.unsqueeze(0).to(emb.dtype), emb[:, len(pre) + L:]], 1)
    pos = torch.arange(ids.size(1), device=M.dev).unsqueeze(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hs = M.base.model(inputs_embeds=emb, position_ids=pos, use_cache=False).last_hidden_state
    n = len(full)
    lg = M.base.lm_head(hs[0, n - 1:n - 1 + nE]).float()
    return lg, ids[0, n:n + nE], emb


def nll(lg, tgt):
    return float(F.cross_entropy(lg, tgt[:lg.size(0)].to(lg.device) if torch.is_tensor(tgt) else torch.tensor(tgt[:lg.size(0)], device=lg.device)))


def item_terms(S, Tm, s, n_tail):
    q, tgt = s["question"], S.target_ids(s["execution"]); nE = len(tgt) - n_tail
    z = S.zlat(q)
    lw, yw, ew = fwd(S, student_pre(S, q), z, tgt, nE)
    with torch.no_grad():
        lb, yb, eb = fwd(S, student_pre(S, q), torch.zeros_like(z), tgt, nE)
        lt, yt, et = fwd(Tm, teacher_pre(Tm, q, s["answer"]), z.detach(), tgt, nE)
    d = jsd(lt.detach(), lw).mean()
    a = jsd(lw, lb.detach()).mean()
    aux = {"nE": nE, "labels_equal": bool(torch.equal(yw, yb) and torch.equal(yw, yt)) and yw.tolist() == tgt[:nE], "z": z, "emb_with": ew, "emb_blind": eb,
           "n_pre_student": len(student_pre(S, q)), "n_pre_teacher": len(teacher_pre(Tm, q, s["answer"]))}
    return d, a, aux


def evaluate(S, Tm, train, ev, n_tail, gen_tokens):
    out = {}
    with torch.no_grad():
        d_tr, a_tr = [], []
        for s in train:
            d, a, _ = item_terms(S, Tm, s, n_tail); d_tr.append(float(d)); a_tr.append(float(a))
        out["train_distill_jsd"] = float(np.mean(d_tr)); out["train_with_blind_jsd"] = float(np.mean(a_tr))
        Z = [S.zlat(s["question"]) for s in ev]; rows = []
        for i, s in enumerate(ev):
            q, tgt = s["question"], S.target_ids(s["execution"]); nE = len(tgt) - n_tail; pre = student_pre(S, q)
            lw, _, _ = fwd(S, pre, Z[i], tgt, nE); lb, _, _ = fwd(S, pre, torch.zeros_like(Z[i]), tgt, nE)
            lx, _, _ = fwd(S, pre, Z[(i + 1) % len(ev)], tgt, nE); ld, _, _ = fwd(S, pre, None, tgt, nE, slots=False)
            y = torch.tensor(tgt[:nE], device=S.dev)
            r = {"id": s["id"], "jsd_with_blind": float(jsd(lw, lb).mean()), "nll_self": nll(lw, y)}
            for nm, lg in (("zero", lb), ("crossq", lx), ("drop", ld)):
                r[f"dnll_{nm}"] = nll(lg, y) - r["nll_self"]
            rows.append(r)
        gens = generate_batch(S, [s["question"] for s in ev], torch.stack(Z), gen_tokens)
    for r, s, g in zip(rows, ev, gens):
        p = last_boxed(g); r.update({"pred": p, "correct": correct(p, str(s["answer"])), "format_ok": "</EXECUTION>" in g and "<|im_end|>" in g, "text": g})
    out["eval_correct"] = int(sum(r["correct"] for r in rows)); out["eval_format_ok"] = int(sum(r["format_ok"] for r in rows)); out["n_eval"] = len(rows)
    for k in ("jsd_with_blind", "nll_self", "dnll_zero", "dnll_crossq", "dnll_drop"):
        out[f"eval_{k}"] = float(np.mean([r[k] for r in rows]))
    out["eval_zs_crossq_cos"] = float(F.cosine_similarity(torch.stack(Z).flatten(1).unsqueeze(1), torch.stack(Z).flatten(1).unsqueeze(0), -1).triu(1).sum() / (len(Z) * (len(Z) - 1) / 2))
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["audit", "train"], required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--s2b", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=10); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--n_train", type=int, default=16)
    ap.add_argument("--gen_tokens", type=int, default=2048)
    ap.add_argument("--lam_anchor", type=float, default=1.0); ap.add_argument("--stop_if_correct_le", type=int, default=-1)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    global LAM_ANCHOR
    LAM_ANCHOR = a.lam_anchor
    ck = torch.load(a.ckpt, weights_only=False)
    print("CHECKPOINT", a.ckpt, "step", ck.get("step"), "phase", ck.get("phase"), "| lam_distill", LAM_DISTILL, "lam_anchor", LAM_ANCHOR, "| lr", a.lr, "steps", a.steps, "n_train", a.n_train, flush=True)
    S = TStudent(a.model, a.s2b, seed=0); S.load(ck["state"])
    Tm = TStudent(a.model, a.s2b, seed=0); Tm.load(ck["state"])
    for p in Tm.peft.parameters():
        p.requires_grad_(False)
    n_tail = len(S.tok.encode("</EXECUTION><|im_end|>", add_special_tokens=False))
    singles = load_jsonl(f"{a.s2b}/data/train_singles.jsonl"); vals = load_jsonl(f"{a.s2b}/data/val_singles.jsonl")
    train, ev = singles[:a.n_train], vals[48:64]                 # fixed IDs; eval disjoint from train and from the 48-question latent set
    params = S.params()
    meta = {"ckpt": a.ckpt, "ckpt_step": ck.get("step"), "train_ids": [s["id"] for s in train], "eval_ids": [s["id"] for s in ev],
            "student_trainable": {"projector": sum(p.numel() for _, p in S.proj), "lora": sum(p.numel() for _, p in S.lora)},
            "teacher_trainable": sum(p.numel() for p in Tm.peft.parameters() if p.requires_grad), "hint_template": HINT, "lam_distill": LAM_DISTILL, "lam_anchor": LAM_ANCHOR}

    if a.mode == "audit":
        g = dict(meta); B = train[:4]
        # values, masks, alignment
        vals_ = []
        for s in B:
            d, an, aux = item_terms(S, Tm, s, n_tail)
            vals_.append({"id": s["id"], "distill_jsd": float(d), "anchor_jsd_pos": float(an), "total": float(LAM_DISTILL * d - LAM_ANCHOR * an), "n_valid_tokens": aux["nE"],
                          "labels_aligned_3_branches": aux["labels_equal"], "n_pre_student": aux["n_pre_student"], "n_pre_teacher": aux["n_pre_teacher"]})
        g["items"] = vals_
        g["finite"] = all(math.isfinite(v["total"]) for v in vals_); g["aligned"] = all(v["labels_aligned_3_branches"] for v in vals_)
        # separate backward per term
        gr = {}
        for nm in ("distill", "anchor", "total"):
            for p in params:
                p.grad = None
            for s in B:
                d, an, _ = item_terms(S, Tm, s, n_tail)
                loss = {"distill": LAM_DISTILL * d, "anchor": -LAM_ANCHOR * an, "total": LAM_DISTILL * d - LAM_ANCHOR * an}[nm]
                (loss / len(B)).backward()
            gr[nm] = {"projector": grad_norm([p for _, p in S.proj]), "lora": grad_norm([p for _, p in S.lora]),
                      "finite": all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)}
        gr["ratio_anchor_over_distill"] = {k: gr["anchor"][k] / max(gr["distill"][k], 1e-30) for k in ("projector", "lora")}
        g["grads"] = gr
        g["teacher_grads_none"] = all(p.grad is None for p in Tm.peft.parameters())
        # latent replacement really changes the with-branch inputs / outputs
        s = B[0]; q, tgt = s["question"], S.target_ids(s["execution"]); nE = len(tgt) - n_tail; pre = student_pre(S, q)
        with torch.no_grad():
            z = S.zlat(q); zx = S.zlat(B[1]["question"])
            lw, _, ew = fwd(S, pre, z, tgt, nE); lz, _, ez = fwd(S, pre, torch.zeros_like(z), tgt, nE); lx, _, ex = fwd(S, pre, zx, tgt, nE)
        sl = slice(len(pre), len(pre) + L)
        g["replace"] = {"slots_hold_zS": bool(torch.equal(ew[0, sl], z.to(ew.dtype))), "zero_changes_slot_emb": float((ew[0, sl] - ez[0, sl]).abs().max()),
                        "crossq_changes_slot_emb": float((ew[0, sl] - ex[0, sl]).abs().max()),
                        "nonslot_emb_identical": bool(torch.equal(ew[0, :len(pre)], ez[0, :len(pre)]) and torch.equal(ew[0, len(pre) + L:], ez[0, len(pre) + L:])),
                        "zero_logit_maxdiff": float((lw - lz).abs().max()), "crossq_logit_maxdiff": float((lw - lx).abs().max()),
                        "zS_vs_crossq_cos": float(F.cosine_similarity(z.flatten(), zx.flatten(), 0)), "jsd_with_zero": float(jsd(lw, lz).mean()), "jsd_with_crossq": float(jsd(lw, lx).mean())}
        # answer only in the teacher prompt
        sd, td = S.tok.decode(student_pre(S, q)), Tm.tok.decode(teacher_pre(Tm, q, s["answer"]))
        g["answer_leak"] = {"answer": s["answer"], "in_student_prompt": ("Reference final answer" in sd), "in_teacher_prompt": (str(s["answer"]) in td and "Reference final answer" in td),
                            "teacher_prompt_tail": td[-160:]}
        hard = [g["finite"], g["aligned"], gr["distill"]["finite"], gr["anchor"]["finite"], gr["total"]["finite"], gr["distill"]["projector"] > 0, gr["anchor"]["projector"] > 0,
                gr["anchor"]["lora"] > 0, meta["teacher_trainable"] == 0, g["teacher_grads_none"], g["replace"]["slots_hold_zS"], g["replace"]["nonslot_emb_identical"],
                g["replace"]["zero_logit_maxdiff"] > 0, not g["answer_leak"]["in_student_prompt"], g["answer_leak"]["in_teacher_prompt"]]
        g["hard_vector"] = hard; g["ALL_HARD_PASS"] = all(hard)
        json.dump(g, open(f"{a.out}/audit.json", "w"), indent=1); print(json.dumps(g, indent=1))
        return

    # ---- train (<= 10 steps), eval at 0 / 5 / 10
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)     # fresh AdamW (new objective)
    os.makedirs(f"{a.out}/checkpoints", exist_ok=True); os.makedirs(f"{a.out}/evaluations", exist_ok=True)
    log = open(f"{a.out}/train.jsonl", "w"); t0 = time.time()
    json.dump(meta, open(f"{a.out}/run_meta.json", "w"), indent=1)

    def do_eval(step):
        res, rows = evaluate(S, Tm, train, ev, n_tail, a.gen_tokens); res["step"] = step
        json.dump(res, open(f"{a.out}/evaluations/eval_step{step}.json", "w"), indent=1)
        with open(f"{a.out}/evaluations/rows_step{step}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("EVAL", json.dumps(res), flush=True)
        return res
    do_eval(0)
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True); rec = []
        for s in train:
            d, an, _ = item_terms(S, Tm, s, n_tail)
            ((LAM_DISTILL * d - LAM_ANCHOR * an) / len(train)).backward(); rec.append((float(d), float(an)))
        gp, gl = grad_norm([p for _, p in S.proj]), grad_norm([p for _, p in S.lora])
        fin = all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0)); opt.step()
        dm, am = float(np.mean([x[0] for x in rec])), float(np.mean([x[1] for x in rec]))
        r = {"step": step, "distill_jsd": dm, "anchor_jsd_pos": am, "total_loss": LAM_DISTILL * dm - LAM_ANCHOR * am, "grad_norm_projector": gp, "grad_norm_lora": gl,
             "grad_total_preclip": gn, "with_vs_blind_jsd": am, "finite": fin, "elapsed_s": time.time() - t0, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
        log.write(json.dumps(r) + "\n"); log.flush(); print("STEP", json.dumps(r), flush=True)
        if not fin or not math.isfinite(r["total_loss"]):
            print("EARLY STOP: non-finite"); break
        if step in (5, a.steps):
            save_ck(S, opt, f"{a.out}/checkpoints/stage2c_step{step}.pt", step, {"init": a.ckpt, "lam_distill": LAM_DISTILL, "lam_anchor": LAM_ANCHOR})
            res = do_eval(step)
            if step < a.steps and res["eval_correct"] <= a.stop_if_correct_le:
                print(f"EARLY STOP: eval_correct {res['eval_correct']} <= {a.stop_if_correct_le} at step {step}"); break


if __name__ == "__main__":
    main()
