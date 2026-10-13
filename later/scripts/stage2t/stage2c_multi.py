"""Stage 2C three-loss vs four-loss (<= 200 local steps) from Stage 2B joint global step 500.

  L3 = 1.0 * Distill + 1.0 * Anchor + 0.1 * Exec_FR                 (--lam_cos 0: the TF branch is never computed)
  L4 = L3 + 1.0 * Cos_TF                                             (--lam_cos 1.0)
Distill = mean_t JSD(sg p_T(.|Q+a*, sg z_S, E<t), p_with)      Anchor = - mean_t JSD(p_with, sg p_blind)   (Anchor already carries the minus)
Exec_FR = mean CE over E + '</EXECUTION><|im_end|>' with the free-running z_S (no detach)   (= joint_2b exec target)
Cos_TF  = joint_2b.cosd(joint_2b.tf_latents(S, Q, Z_T), Z_T)                                 (original Stage 2B TF branch, imported)
One z_S per item (graph kept) shared by Distill / Anchor / Exec; one with-branch forward over the full target (rows [:nE] -> JSD,
all rows -> CE); blind / teacher forwards from anchorsim_2c.fwd (no_grad). Teacher = frozen copy of the init + answer hint (no EMA).
Optimizer / data / checks identical to stage2c_long (fresh AdamW lr 1e-4 wd 0, clip 1.0, bf16, seed 0, 16/step, batches(seed=42)).
Modes: audit (0-step, per-term separate backward, no optimizer step) | train (smoke checks at 5/10 on 16 unseen questions incl. full
generation, light checks at 50/100/150/200, 48-q latent diag at 100, checkpoints at 0/10/50/100/150/200).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from later.scripts.stage2s.stochastic_pilot import grad_norm
from later.scripts.stage2t.anchorsim_2c import HINT, fwd, item_terms, jsd, student_pre, teacher_pre
from later.scripts.stage2t.joint_2b import cosd, tf_latents
from later.scripts.stage2t.joint_2b_continue import diag
from later.scripts.stage2t.stage2c_long import INIT_GLOBAL_STEP, light, sha256
from later.scripts.stage2t.three_step import TStudent, batches, load_jsonl, save_ck

LAM_DISTILL, LAM_ANCHOR, LAM_EXEC = 1.0, 1.0, 0.1


def item(S, Tm, s, n_tail, lam_cos):
    q, tgt = s["question"], S.target_ids(s["execution"]); nE = len(tgt) - n_tail; pre = student_pre(S, q)
    z = S.zlat(q)
    lw_full, yw, _ = fwd(S, pre, z, tgt, len(tgt))
    lw = lw_full[:nE]
    with torch.no_grad():
        lb, yb, _ = fwd(S, pre, torch.zeros_like(z), tgt, nE)
        lt, yt, _ = fwd(Tm, teacher_pre(Tm, q, s["answer"]), z.detach(), tgt, nE)
    parts = {"distill": jsd(lt.detach(), lw).mean(), "anchor": -jsd(lw, lb.detach()).mean(),
             "exec": F.cross_entropy(lw_full, torch.tensor(tgt, device=S.dev))}
    if lam_cos > 0:
        T = S.teacher(s["plan_norm"]); parts["cos"] = cosd(tf_latents(S, q, T), T)
    info = {"nE": nE, "n_tgt": len(tgt), "aligned": bool(torch.equal(yw[:nE], yb) and torch.equal(yb, yt)) and yw.tolist() == tgt}
    return parts, info


def total(parts, lam_cos):
    t = LAM_DISTILL * parts["distill"] + LAM_ANCHOR * parts["anchor"] + LAM_EXEC * parts["exec"]
    return t + lam_cos * parts["cos"] if lam_cos > 0 else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["audit", "train"], required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--s2b", required=True); ap.add_argument("--out", required=True); ap.add_argument("--init", required=True)
    ap.add_argument("--lam_cos", type=float, required=True)
    ap.add_argument("--steps", type=int, default=200); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--n_single", type=int, default=16)
    ap.add_argument("--save_at", default="0,10,50,100,150,200"); ap.add_argument("--smoke_at", default="0,5,10"); ap.add_argument("--light_at", default="50,100,150,200")
    ap.add_argument("--diag_at", default="100"); ap.add_argument("--gen_tokens", type=int, default=2048); ap.add_argument("--data_seed", type=int, default=42)
    ap.add_argument("--student_ckpt", default=None, help="audit only: load the Student from this Stage 2C checkpoint (Teacher stays the init)")
    a = ap.parse_args()
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
    ev = vals[48:64]
    params = S.params()
    lams = {"distill": LAM_DISTILL, "anchor": LAM_ANCHOR, "exec": LAM_EXEC, "cos": a.lam_cos}
    meta = {"mode": a.mode, "init": a.init, "init_sha256": sha256(a.init), "init_global_step": ck["step"], "init_phase": ck.get("phase"),
            "gpu": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "SLURM_STEP_GPUS": os.environ.get("SLURM_STEP_GPUS"), "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
                    "name": torch.cuda.get_device_name(0), "uuid": str(torch.cuda.get_device_properties(0).uuid)},
            "lambdas": lams, "loss": "1.0*JSD(sg pT, p_with) - 1.0*JSD(p_with, sg p_blind) + 0.1*CE_FR(E+close) " + ("+ 1.0*TF cos (joint_2b)" if a.lam_cos > 0 else "(TF cos not computed)"),
            "ema": False, "hint_template": HINT, "z_teacher_source": f"{a.s2b}/data/z_cache.pt (frozen Stage 2A step-300 compressor outputs, read-only)",
            "optimizer": f"AdamW(lr={a.lr}, wd=0, fresh), constant lr, clip 1.0", "precision": "bf16 autocast, fp32 JSD / CE / cos", "seed": 0,
            "batch": f"{a.n_single}/step, per-sample backward /{a.n_single}", "data": f"{a.s2b}/data/train_singles.jsonl", "data_sha256": sha256(f"{a.s2b}/data/train_singles.jsonl"),
            "n_train": len(singles), "data_order": f"three_step.batches(seed={a.data_seed})", "smoke_eval_ids": [s["id"] for s in ev], "light_gen_ids": [s["id"] for s in ev[:4]],
            "student_trainable": {"projector": sum(p.numel() for _, p in S.proj), "lora": sum(p.numel() for _, p in S.lora), "n_tensors": [len(S.proj), len(S.lora)]},
            "student_frozen_numel": sum(p.numel() for p in S.peft.parameters() if not p.requires_grad), "teacher_trainable": sum(p.numel() for p in Tm.peft.parameters() if p.requires_grad),
            "teacher_snapshot_tensors": len(t_snap)}
    assert meta["teacher_trainable"] == 0
    print("META", json.dumps(meta), flush=True)

    if a.mode == "audit":
        tag = ""
        if a.student_ckpt:
            sc = torch.load(a.student_ckpt, weights_only=False); S.load(sc["state"]); tag = f"_step{sc['step']}"
            meta["student_ckpt"] = a.student_ckpt; meta["student_ckpt_local_step"] = sc["step"]
        B = singles[:4]; g = {"meta": meta, "items": []}
        with torch.no_grad():
            for s in B:
                parts, info = item(S, Tm, s, n_tail, a.lam_cos)
                d0, a0, aux0 = item_terms(S, Tm, s, n_tail)                       # reference implementation used by all earlier 2C runs
                sf, _, _ = S.score(s["question"], S.zlat(s["question"]).unsqueeze(0), [S.target_ids(s["execution"])])   # reference 2B exec CE path
                g["items"].append({"id": s["id"], **{k: float(v) for k, v in parts.items()}, "total": float(total(parts, a.lam_cos)), **info,
                                   "ref_distill": float(d0), "ref_anchor_pos": float(a0), "ref_exec_score": float(-sf[0]), "ref_aligned": aux0["labels_equal"]})
        it = g["items"]
        g["finite"] = all(math.isfinite(x["total"]) for x in it); g["aligned"] = all(x["aligned"] and x["ref_aligned"] for x in it)
        g["max_abs_diff_vs_ref"] = {"distill": max(abs(x["distill"] - x["ref_distill"]) for x in it), "anchor": max(abs(-x["anchor"] - x["ref_anchor_pos"]) for x in it),
                                    "exec": max(abs(x["exec"] - x["ref_exec_score"]) for x in it)}
        g["cos_computed"] = "cos" in it[0]
        gr = {}
        for nm in ["distill", "anchor", "exec"] + (["cos"] if a.lam_cos > 0 else []) + ["total"]:
            for p in params:
                p.grad = None
            for s in B:
                parts, _ = item(S, Tm, s, n_tail, a.lam_cos)
                loss = total(parts, a.lam_cos) if nm == "total" else lams[nm] * parts[nm]
                (loss / len(B)).backward()
            gr[nm + ("" if nm == "total" else "_weighted")] = {"projector": grad_norm([p for _, p in S.proj]), "lora": grad_norm([p for _, p in S.lora]),
                                                               "finite": all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None),
                                                               "lambda": lams.get(nm, 1.0)}
        for nm, v in list(gr.items()):
            if nm != "total" and v["lambda"]:
                gr[nm.replace("_weighted", "_raw")] = {k: v[k] / v["lambda"] for k in ("projector", "lora")}
        g["grads"] = gr; g["teacher_grads_none"] = all(p.grad is None for p in Tm.peft.parameters())
        hard = {"finite": g["finite"], "aligned": g["aligned"], "matches_ref_distill": g["max_abs_diff_vs_ref"]["distill"] < 1e-4,
                "matches_ref_anchor": g["max_abs_diff_vs_ref"]["anchor"] < 1e-4, "matches_ref_exec": g["max_abs_diff_vs_ref"]["exec"] < 5e-3,
                "all_grads_finite": all(v.get("finite", True) for v in gr.values()), "exec_reaches_projector": gr["exec_weighted"]["projector"] > 0,
                "with_branch_reaches_projector": gr["distill_weighted"]["projector"] > 0 and gr["anchor_weighted"]["projector"] > 0,
                "cos_only_if_lam_cos": g["cos_computed"] == (a.lam_cos > 0), "teacher_frozen": meta["teacher_trainable"] == 0 and g["teacher_grads_none"]}
        if a.lam_cos > 0:
            hard["cos_reaches_projector"] = gr["cos_weighted"]["projector"] > 0
        g["hard"] = hard; g["ALL_HARD_PASS"] = all(hard.values())
        json.dump(g, open(f"{a.out}/audit_terms{tag}.json", "w"), indent=1); print(json.dumps(g, indent=1))
        return

    json.dump(meta, open(f"{a.out}/run_meta.json", "w"), indent=1)
    allB = batches(singles, [], a.n_single, 0, a.steps, seed=a.data_seed)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    S_ = lambda x: {int(v) for v in x.split(",") if v}
    save_at, smoke_at, light_at, diag_at = S_(a.save_at), S_(a.smoke_at), S_(a.light_at), S_(a.diag_at)
    log = open(f"{a.out}/train.jsonl", "a"); t0 = time.time(); state = {"prev_severe": False, "smoke0": None}

    def checkpoint(step):
        save_ck(S, opt, f"{a.out}/checkpoints/stage2c_step{step}.pt", step, {"phase": "stage2c", "stage2c_local_step": step, "init": a.init,
                "init_global_step_2b": INIT_GLOBAL_STEP, "lambdas": lams, "lr": a.lr})

    def checks(step):
        smoke = step in smoke_at; tc = time.time()
        res, rows = light(S, Tm, ev, n_tail, len(ev) if smoke else 4, a.gen_tokens, teacher_too=(step == 0)); res["step"] = step; res["kind"] = "smoke16" if smoke else "light4"
        res["teacher_unchanged"] = teacher_unchanged()
        if step in diag_at:
            d = diag(S, vals, 48, f"stage2c local step {step}"); json.dump(d, open(f"{a.out}/evaluations/diag_stage2c_step{step}.json", "w"), indent=1)
            res["diag48"] = {"fr_paired": d["fr"]["paired"], "fr_gap_perm": d["fr"]["gap_perm"]["mean"], "fr_gap_ci": d["fr"]["gap_perm"]["ci95"], "fr_gap_frac_pos": d["fr"]["gap_perm"]["frac_pos"],
                             "zs_crossq_cos": d["collapse"]["zs_crossq_cos"], "var_ratio": d["collapse"]["zs_var_over_teacher_var"]}
        stop = None
        if smoke:
            if step == 0:
                state["smoke0"] = res
            else:                     # pre-registered smoke criteria vs this run's own step 0 (same 16 questions, full generation)
                b = state["smoke0"]
                res["smoke_collapse"] = (res["gen_format_ok"] <= b["gen_format_ok"] - 4 or res["gen_correct"] <= b["gen_correct"] // 2
                                         or res["gen_empty"] >= 3 or res["gen_degenerate"] >= 3)
                if res["smoke_collapse"]:
                    stop = f"smoke collapse at step {step}"
        else:
            if res["severe_collapse"] and state["prev_severe"]:
                stop = "two consecutive 4-question checks with severe collapse"
            state["prev_severe"] = res["severe_collapse"]
        if not res["teacher_unchanged"]:
            stop = "teacher parameters changed"
        res["check_seconds"] = time.time() - tc
        json.dump(res, open(f"{a.out}/evaluations/light_step{step}.json", "w"), indent=1)
        with open(f"{a.out}/evaluations/light_rows_step{step}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("CHECK", json.dumps(res), flush=True)
        return stop

    if 0 in save_at:
        checkpoint(0)
    if (stop := checks(0)):
        print("STOP:", stop); return
    for step in range(1, a.steps + 1):
        ts = time.time(); si, _ = allB[step - 1]; n = len(si)
        opt.zero_grad(set_to_none=True); rec = []; ntok = 0; aligned = True
        for i in si:
            parts, info = item(S, Tm, singles[i], n_tail, a.lam_cos)
            (total(parts, a.lam_cos) / n).backward(); rec.append({k: float(v) for k, v in parts.items()}); ntok += info["n_tgt"]; aligned &= info["aligned"]
        gp, gl = grad_norm([p for _, p in S.proj]), grad_norm([p for _, p in S.lora])
        fin = all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)
        t_nograd = all(p.grad is None for p in Tm.peft.parameters())
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0)); opt.step()
        m = {k: float(np.mean([r[k] for r in rec])) for k in rec[0]}; dt = time.time() - ts
        tot = sum(lams[k] * v for k, v in m.items())
        r = {"step": step, "distill_jsd": m["distill"], "anchor_jsd_pos": -m["anchor"], "exec_ce": m["exec"], "cos_tf": m.get("cos"), "total_loss": tot,
             "grad_norm_projector": gp, "grad_norm_lora": gl, "grad_total_preclip": gn, "clipped": gn > 1.0, "lr": opt.param_groups[0]["lr"], "finite": fin,
             "labels_aligned": aligned, "teacher_grad_none": t_nograd, "exec_tokens": ntok, "step_s": dt, "elapsed_s": time.time() - t0,
             "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "batch": si, "epoch": (step - 1) * n / len(singles)}
        log.write(json.dumps(r) + "\n"); log.flush()
        print(f"step {step} distill {m['distill']:.6f} anchor {-m['anchor']:.6f} exec {m['exec']:.4f} cos {m.get('cos', float('nan')):.4f} total {tot:+.5f} "
              f"gn {gn:.4f} (p {gp:.4f} l {gl:.4f}) fin {fin} {dt:.1f}s {r['elapsed_s']:.0f}s", flush=True)
        if not fin or not math.isfinite(tot) or not aligned or not t_nograd:
            print("STOP: non-finite / misaligned / teacher grad"); checkpoint(f"{step}_stopped"); break
        if step in save_at:
            checkpoint(step)
        if (step in smoke_at or step in light_at) and (stop := checks(step)):
            print("STOP:", stop); break


if __name__ == "__main__":
    main()
