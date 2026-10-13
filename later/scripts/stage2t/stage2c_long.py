"""Stage 2C long run (<= 200 local steps): the ORIGINAL fixed-Teacher two-JSD objective, unchanged, from Stage 2B joint global step 500.

Objective / forward code is imported verbatim from anchorsim_2c (item_terms, fwd, jsd, teacher prompt with the answer hint):
  L = 1.0 * mean_t JSD(sg p_T, p_with) - 0.1 * mean_t JSD(p_with, sg p_blind)
Student = joint_step500 (LoRA + projector trainable, base / 2A frozen). Teacher = a second, fully frozen copy of the SAME checkpoint
(sees a* in its user turn and sg(z_S)). No EMA, no cos / CE / bridge terms.
Same optimizer settings as the 10-step lambda=0.1 run: fresh AdamW lr 1e-4 wd 0 (constant), clip 1.0, bf16 autocast, seed 0,
16 samples per step (per-sample backward / 16). Only change of setting vs that run: data = all 2000 Stage 2B train_singles,
ordered by the repo's existing 2C order batches(singles, [], 16, 0, steps, seed=42) instead of the same 16 samples every step.
Local steps 0..200 (NOT Stage 2B global steps). Checkpoints at --save_at (with AdamW + RNG).
Light checks at --light_at: 16 fixed val questions teacher-forced (with/blind JSD, zero/swap/drop dNLL = NLL(alt) - NLL(with)),
4 of them free-generated (accuracy / format / collapse); + the 48-question latent diag (joint_2b_continue.diag) at --diag_at.
"""
import argparse
import hashlib
import json
import math
import os
import time
import zlib

import numpy as np
import torch

from later.scripts.bridge.mini_bridge import correct, last_boxed
from later.scripts.stage2s.stochastic_pilot import grad_norm
from later.scripts.stage2t.anchorsim_2c import HINT, fwd, item_terms, jsd, nll, student_pre, teacher_pre
from later.scripts.stage2t.contrastive_2b import generate_batch
from later.scripts.stage2t.joint_2b_continue import diag
from later.scripts.stage2t.three_step import TStudent, batches, load_jsonl, save_ck

LAM_DISTILL, LAM_ANCHOR = 1.0, 0.1
INIT_GLOBAL_STEP = 500


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def degenerate(t):
    """crude repetition flag: long text that compresses extremely well."""
    return len(t) > 800 and len(zlib.compress(t.encode())) / len(t.encode()) < 0.08


def light(S, Tm, ev, n_tail, n_gen, gen_tokens, teacher_too=False):
    out, rows = {}, []
    with torch.no_grad():
        Z = [S.zlat(s["question"]) for s in ev]
        for i, s in enumerate(ev):
            q, tgt = s["question"], S.target_ids(s["execution"]); nE = len(tgt) - n_tail; pre = student_pre(S, q)
            lw, _, _ = fwd(S, pre, Z[i], tgt, nE); lb, _, _ = fwd(S, pre, torch.zeros_like(Z[i]), tgt, nE)
            lx, _, _ = fwd(S, pre, Z[(i + 1) % len(ev)], tgt, nE); ld, _, _ = fwd(S, pre, None, tgt, nE, slots=False)
            y = torch.tensor(tgt[:nE], device=S.dev)
            r = {"id": s["id"], "nE": nE, "jsd_with_blind": float(jsd(lw, lb).mean()), "jsd_with_swap": float(jsd(lw, lx).mean()), "nll_self": nll(lw, y)}
            for nm, lg in (("zero", lb), ("swap", lx), ("drop", ld)):
                r[f"dnll_{nm}"] = nll(lg, y) - r["nll_self"]
            if teacher_too:          # the frozen teacher's own latent dependence (answer-conditioned prompt, same z_S / zeros)
                tp = teacher_pre(Tm, q, s["answer"])
                ltw, _, _ = fwd(Tm, tp, Z[i], tgt, nE); ltb, _, _ = fwd(Tm, tp, torch.zeros_like(Z[i]), tgt, nE)
                r["teacher_jsd_with_blind"] = float(jsd(ltw, ltb).mean()); r["teacher_dnll_zero"] = nll(ltb, y) - nll(ltw, y)
                r["distill_jsd"] = float(jsd(ltw, lw).mean())
            rows.append(r)
        gens = generate_batch(S, [s["question"] for s in ev[:n_gen]], torch.stack(Z[:n_gen]), gen_tokens)
    for r, s, g in zip(rows, ev, gens):
        p = last_boxed(g)
        r.update({"pred": p, "correct": correct(p, str(s["answer"])), "format_ok": "</EXECUTION>" in g and "<|im_end|>" in g,
                  "empty": len(g.replace("</EXECUTION>", "").replace("<|im_end|>", "").strip()) < 20, "degenerate": degenerate(g), "gen_chars": len(g), "text": g})
    keys = ["jsd_with_blind", "jsd_with_swap", "nll_self", "dnll_zero", "dnll_swap", "dnll_drop"] + (["teacher_jsd_with_blind", "teacher_dnll_zero", "distill_jsd"] if teacher_too else [])
    for k in keys:
        v = [r[k] for r in rows]; out[k] = float(np.mean(v)); out[k + "_frac_pos"] = float(np.mean(np.array(v) > 0))
    G = rows[:n_gen]
    out.update({"n_tf": len(rows), "n_gen": len(G), "gen_correct": int(sum(r["correct"] for r in G)), "gen_format_ok": int(sum(r["format_ok"] for r in G)),
                "gen_empty": int(sum(r["empty"] for r in G)), "gen_degenerate": int(sum(r["degenerate"] for r in G))})
    out["severe_collapse"] = out["gen_format_ok"] <= 1 or out["gen_empty"] >= 2 or out["gen_degenerate"] >= 2
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--s2b", required=True); ap.add_argument("--out", required=True); ap.add_argument("--init", required=True)
    ap.add_argument("--steps", type=int, default=200); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--n_single", type=int, default=16)
    ap.add_argument("--save_at", default="0,50,100,150,200"); ap.add_argument("--light_at", default="0,50,100,150,200"); ap.add_argument("--diag_at", default="100")
    ap.add_argument("--n_gen", type=int, default=4); ap.add_argument("--gen_tokens", type=int, default=2048); ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--lam_anchor", type=float, default=0.1)
    a = ap.parse_args()
    global LAM_ANCHOR
    LAM_ANCHOR = a.lam_anchor
    for d in ("checkpoints", "evaluations"):
        os.makedirs(f"{a.out}/{d}", exist_ok=True)
    ck = torch.load(a.init, weights_only=False)
    assert ck["step"] == INIT_GLOBAL_STEP and ck.get("phase") == "joint", "init must be Stage 2B joint global step 500"
    S = TStudent(a.model, a.s2b, seed=0); S.load(ck["state"])
    Tm = TStudent(a.model, a.s2b, seed=0); Tm.load(ck["state"])
    for p in Tm.peft.parameters():
        p.requires_grad_(False)
    t_snap = {n: p.detach().clone() for n, p in Tm.peft.named_parameters() if "lora_" in n or "latent_projector" in n}
    teacher_unchanged = lambda: all(torch.equal(p, t_snap[n]) for n, p in Tm.peft.named_parameters() if n in t_snap)
    n_tail = len(S.tok.encode("</EXECUTION><|im_end|>", add_special_tokens=False))
    singles = load_jsonl(f"{a.s2b}/data/train_singles.jsonl"); vals = load_jsonl(f"{a.s2b}/data/val_singles.jsonl")
    ev = vals[48:64]                                    # = the fixed 16 eval questions of the earlier 2C runs; disjoint from the 48-q diag set
    params = S.params()
    allB = batches(singles, [], a.n_single, 0, a.steps, seed=a.data_seed)
    seen = [i for si, _ in allB for i in si]
    frozen = sum(p.numel() for p in S.peft.parameters() if not p.requires_grad)
    meta = {"init": a.init, "init_sha256": sha256(a.init), "init_global_step": ck["step"], "gpu": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "SLURM_JOB_GPUS": os.environ.get("SLURM_JOB_GPUS"), "SLURM_STEP_GPUS": os.environ.get("SLURM_STEP_GPUS"), "name": torch.cuda.get_device_name(0), "uuid": str(torch.cuda.get_device_properties(0).uuid)}, "init_phase": ck.get("phase"), "init_meta": {k: ck.get(k) for k in ("lam_cos", "lam_exec", "lr", "resumed_from")},
            "stage2c_local_steps": a.steps, "lam_distill": LAM_DISTILL, "lam_anchor": LAM_ANCHOR, "ema": False, "teacher": "frozen copy of init, answer hint in user turn",
            "hint_template": HINT, "optimizer": f"AdamW(lr={a.lr}, wd=0, fresh), constant lr, clip 1.0", "precision": "bf16 autocast, fp32 log_softmax JSD", "seed": 0,
            "batch": f"{a.n_single} samples/step, per-sample backward /{a.n_single}", "data": f"{a.s2b}/data/train_singles.jsonl", "data_sha256": sha256(f"{a.s2b}/data/train_singles.jsonl"),
            "n_train": len(singles), "n_unique_ids": len({s['id'] for s in singles}), "data_order": f"three_step.batches(seed={a.data_seed})", "samples_seen": len(seen),
            "unique_samples_seen": len(set(seen)), "epochs": len(seen) / len(singles), "first_batch": allB[0][0],
            "light_eval_ids": [s["id"] for s in ev], "light_gen_ids": [s["id"] for s in ev[:a.n_gen]], "gen_tokens": a.gen_tokens, "decoding": "greedy (generate_batch)",
            "student_trainable": {"projector": sum(p.numel() for _, p in S.proj), "lora": sum(p.numel() for _, p in S.lora),
                                  "n_tensors_projector": len(S.proj), "n_tensors_lora": len(S.lora), "lora_names_sample": [n for n, _ in S.lora[:4]],
                                  "projector_names": [n for n, _ in S.proj]},
            "student_frozen_numel": frozen, "teacher_snapshot_tensors": len(t_snap), "teacher_trainable": sum(p.numel() for p in Tm.peft.parameters() if p.requires_grad)}
    json.dump(meta, open(f"{a.out}/run_meta.json", "w"), indent=1); print("META", json.dumps(meta), flush=True)
    assert meta["teacher_trainable"] == 0

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    save_at = {int(x) for x in a.save_at.split(",") if x}; light_at = {int(x) for x in a.light_at.split(",") if x}; diag_at = {int(x) for x in a.diag_at.split(",") if x}
    log = open(f"{a.out}/train.jsonl", "a"); t0 = time.time(); prev_severe = False

    def checkpoint(step):
        save_ck(S, opt, f"{a.out}/checkpoints/stage2c_step{step}.pt", step, {"phase": "stage2c", "stage2c_local_step": step, "init": a.init,
                "init_global_step_2b": INIT_GLOBAL_STEP, "lam_distill": LAM_DISTILL, "lam_anchor": LAM_ANCHOR, "lr": a.lr})

    def checks(step):
        nonlocal prev_severe
        tc = time.time(); res, rows = light(S, Tm, ev, n_tail, a.n_gen, a.gen_tokens, teacher_too=(step == 0)); res["step"] = step
        res["teacher_unchanged"] = teacher_unchanged()
        if step in diag_at:
            d = diag(S, vals, 48, f"stage2c local step {step}"); json.dump(d, open(f"{a.out}/evaluations/diag_stage2c_step{step}.json", "w"), indent=1)
            res["diag48"] = {"fr_paired": d["fr"]["paired"], "fr_gap_perm": d["fr"]["gap_perm"]["mean"], "fr_gap_ci": d["fr"]["gap_perm"]["ci95"], "fr_gap_frac_pos": d["fr"]["gap_perm"]["frac_pos"],
                             "zs_crossq_cos": d["collapse"]["zs_crossq_cos"], "var_ratio": d["collapse"]["zs_var_over_teacher_var"]}
        res["check_seconds"] = time.time() - tc
        json.dump(res, open(f"{a.out}/evaluations/light_step{step}.json", "w"), indent=1)
        with open(f"{a.out}/evaluations/light_rows_step{step}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("LIGHT", json.dumps(res), flush=True)
        stop = None
        if not res["teacher_unchanged"]:
            stop = "teacher parameters changed"
        elif res["severe_collapse"] and prev_severe:
            stop = "two consecutive 4-question checks with severe collapse"
        prev_severe = res["severe_collapse"]
        return stop

    if 0 in save_at:
        checkpoint(0)
    if 0 in light_at and (stop := checks(0)):
        print("STOP:", stop); return
    for step in range(1, a.steps + 1):
        ts = time.time(); si, _ = allB[step - 1]; n = len(si)
        opt.zero_grad(set_to_none=True); rec = []; ntok = 0; aligned = True
        for i in si:
            d, an, aux = item_terms(S, Tm, singles[i], n_tail)
            ((LAM_DISTILL * d - LAM_ANCHOR * an) / n).backward(); rec.append((float(d), float(an))); ntok += aux["nE"]; aligned &= aux["labels_equal"]
        gp, gl = grad_norm([p for _, p in S.proj]), grad_norm([p for _, p in S.lora])
        fin = all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)
        t_nograd = all(p.grad is None for p in Tm.peft.parameters())
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0)); opt.step()
        dm, am = float(np.mean([x[0] for x in rec])), float(np.mean([x[1] for x in rec])); dt = time.time() - ts
        r = {"step": step, "distill_jsd": dm, "anchor_jsd_pos": am, "anchor_weighted": LAM_ANCHOR * am, "total_loss": LAM_DISTILL * dm - LAM_ANCHOR * am,
             "grad_norm_projector": gp, "grad_norm_lora": gl, "grad_total_preclip": gn, "clipped": gn > 1.0, "lr": opt.param_groups[0]["lr"], "finite": fin,
             "labels_aligned": aligned, "teacher_grad_none": t_nograd, "exec_tokens": ntok, "step_s": dt, "elapsed_s": time.time() - t0,
             "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "batch": si, "epoch": (step - 1) * n / len(singles)}
        log.write(json.dumps(r) + "\n"); log.flush()
        print(f"step {step} distill {dm:.6f} anchor {am:.6f} total {r['total_loss']:+.6f} gn {gn:.4f} (p {gp:.4f} l {gl:.4f}) fin {fin} {dt:.1f}s {r['elapsed_s']:.0f}s", flush=True)
        if not fin or not math.isfinite(r["total_loss"]) or not aligned or not t_nograd:
            print("STOP: non-finite / misaligned / teacher grad"); checkpoint(f"{step}_stopped"); break
        if step in save_at:
            checkpoint(step)
        if step in light_at and (stop := checks(step)):
            print("STOP:", stop); break


if __name__ == "__main__":
    main()
