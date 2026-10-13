"""Stage 2B joint two-loss: strict continuation of joint_2b.py from its step-50 checkpoint to global step 500.

Nothing about the objective changes: item_loss / tf_latents are imported from joint_2b (L = 1.0 * TF cos + 0.1 * FR exec CE),
same TStudent, same params() order, AdamW lr 1e-4 wd 0 (constant, no scheduler / warmup in the original), clip 1.0,
same data order batches(singles, [], 16, 0, *, seed=7) indexed by GLOBAL step (the order generator is prefix-stable).
Resume restores projector + LoRA + AdamW state + torch/cuda RNG and asserts ckpt step == 50 and AdamW step == 50.
Modes:
  train : global steps 51..500; saves --save_at; runs the light latent diagnostic (`diag`) in-process at --diag_at.
  diag  : 48 fixed unseen val questions, 8 latents only (no CoT): FR (and TF) paired / shuffled / gap, cross-question cos,
          variance ratio vs teacher, per-slot variance and norms.  Shuffled = roll-1 (historical) and mean over 20 derangements.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from later.scripts.bridge.bridge_input_abcd import boot
from later.scripts.stage2s.stochastic_pilot import grad_norm
from later.scripts.stage2t.joint_2b import LAM_COS, LAM_EXEC, item_loss, tf_latents
from later.scripts.stage2t.three_step import TStudent, batches, load_jsonl, save_ck

START = 50


def diag(S, vals, n_val, ckpt_name, n_perm=20):
    V = vals[:n_val]
    with torch.no_grad():
        T = torch.stack([S.teacher(v["plan_norm"]) for v in V])
        Ptf = torch.stack([tf_latents(S, v["question"], T[i]) for i, v in enumerate(V)])
        Zs = torch.stack([S.zlat(v["question"]) for v in V]).float()
    n = len(V); c8 = lambda X, Y: F.cosine_similarity(X, Y, dim=-1)          # [n, 8]
    rng = random.Random(0); perms = []
    while len(perms) < n_perm:
        p = list(range(n)); rng.shuffle(p)
        if all(p[i] != i for i in range(n)):
            perms.append(torch.tensor(p))
    roll = torch.roll(torch.arange(n), 1)
    st = lambda v: {"mean": float(np.mean(v)), "ci95": boot(list(map(float, v))), "frac_pos": float(np.mean(np.array(v) > 0)),
                    "p10_p50_p90": [float(x) for x in np.percentile(v, [10, 50, 90])]}
    out = {"ckpt": ckpt_name, "n": n, "n_perm": n_perm, "teacher_mean_cos_baseline": float(c8(T.mean(0, keepdim=True).expand_as(T), T).mean())}
    for nm, X in (("fr", Zs), ("tf", Ptf)):
        pa = c8(X, T)                                                        # [n, 8]
        sr = c8(X, T[roll])
        sp = torch.stack([c8(X, T[p]) for p in perms]).mean(0)               # [n, 8] mean over permutations
        out[nm] = {"paired": float(pa.mean()), "shuffled_roll1": float(sr.mean()), "shuffled_perm": float(sp.mean()),
                   "gap_roll1": st((pa - sr).mean(-1).tolist()), "gap_perm": st((pa - sp).mean(-1).tolist()),
                   "paired_per_slot": [float(x) for x in pa.mean(0)], "gap_perm_per_slot": [float(x) for x in (pa - sp).mean(0)],
                   "per_sample_paired": pa.mean(-1).tolist(), "per_sample_shuffled_perm": sp.mean(-1).tolist()}
    fl = Zs.flatten(1); cc = F.cosine_similarity(fl.unsqueeze(1), fl.unsqueeze(0), -1); iu = torch.triu_indices(n, n, 1)
    tl = T.flatten(1); ct = F.cosine_similarity(tl.unsqueeze(1), tl.unsqueeze(0), -1)
    out["collapse"] = {"zs_crossq_cos": float(cc[iu[0], iu[1]].mean()), "teacher_crossq_cos": float(ct[iu[0], iu[1]].mean()),
                       "zs_crossq_cos_per_slot": [float(F.cosine_similarity(Zs[:, k].unsqueeze(1), Zs[:, k].unsqueeze(0), -1)[iu[0], iu[1]].mean()) for k in range(8)],
                       "zs_var_over_teacher_var": float(Zs.var(0).mean() / T.var(0).mean()),
                       "zs_var_over_teacher_var_per_slot": [float(x) for x in Zs.var(0).mean(-1) / T.var(0).mean(-1)],
                       "zs_var_per_slot": [float(x) for x in Zs.var(0).mean(-1)], "teacher_var_per_slot": [float(x) for x in T.var(0).mean(-1)],
                       "zs_norm_per_slot": [float(x) for x in Zs.norm(dim=-1).mean(0)], "teacher_norm_per_slot": [float(x) for x in T.norm(dim=-1).mean(0)],
                       "zs_adjacent_slot_cos": float(F.cosine_similarity(Zs[:, 1:], Zs[:, :-1], -1).mean()),
                       "teacher_adjacent_slot_cos": float(F.cosine_similarity(T[:, 1:], T[:, :-1], -1).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "diag"], required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--s2b", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None, help="joint_2b step-50 checkpoint (projector + LoRA + AdamW + RNG)")
    ap.add_argument("--total_steps", type=int, default=500); ap.add_argument("--n_single", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save_at", default="100,150,200,300,400,500"); ap.add_argument("--diag_at", default="200,400")
    ap.add_argument("--ckpt", default=None); ap.add_argument("--tag", default=None); ap.add_argument("--n_val", type=int, default=48)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/evaluations", exist_ok=True)
    S = TStudent(a.model, a.s2b, seed=0)
    singles = load_jsonl(f"{a.s2b}/data/train_singles.jsonl"); vals = load_jsonl(f"{a.s2b}/data/val_singles.jsonl")
    params = S.params()

    if a.mode == "diag":
        S.load(torch.load(a.ckpt, weights_only=False)["state"])
        res = diag(S, vals, a.n_val, a.ckpt)
        name = a.tag or os.path.basename(a.ckpt).replace(".pt", "")
        json.dump(res, open(f"{a.out}/evaluations/diag_{name}.json", "w"), indent=1)
        print(json.dumps({k: v for k, v in res.items() if k not in ("fr", "tf")} | {k: {kk: vv for kk, vv in res[k].items() if not kk.startswith("per_sample")} for k in ("fr", "tf")}, indent=1))
        return

    # ---- train: strict resume ----
    ck = torch.load(a.init, weights_only=False)
    assert ck["step"] == START and ck.get("phase") == "joint" and ck["lam_cos"] == LAM_COS and ck["lam_exec"] == LAM_EXEC and ck["lr"] == a.lr, "not the joint step-50 ckpt"
    S.load(ck["state"])
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    opt.load_state_dict(ck["opt"])
    adam_steps = sorted({float(s["step"]) for s in opt.state.values()})
    shapes_ok = all(opt.state[p]["exp_avg"].shape == p.shape for p in params)
    torch.set_rng_state(ck["torch_rng"]); torch.cuda.set_rng_state(ck["cuda_rng"])
    allB = batches(singles, [], a.n_single, 0, a.total_steps, seed=7)
    ref50 = batches(singles, [], a.n_single, 0, 50, seed=7)
    res_info = {"init": a.init, "ckpt_step": ck["step"], "adamw_state_steps": adam_steps, "adamw_state_shapes_match_params": shapes_ok,
                "n_params_with_state": len(opt.state), "n_params": len(params), "lr_groups": [g["lr"] for g in opt.param_groups],
                "scheduler": "none (constant lr in joint_2b.py)", "rng_restored": True,
                "data_order_prefix_identical_to_original_50": [b[0] for b in allB[:50]] == [b[0] for b in ref50],
                "n_singles": len(singles), "steps_per_epoch": len(singles) / a.n_single}
    assert adam_steps == [float(START)] and shapes_ok and res_info["data_order_prefix_identical_to_original_50"]
    json.dump(res_info, open(f"{a.out}/resume_audit.json", "w"), indent=1); print(json.dumps(res_info, indent=1), flush=True)
    save_at = {int(x) for x in a.save_at.split(",")}; diag_at = {int(x) for x in a.diag_at.split(",") if x}
    os.makedirs(f"{a.out}/checkpoints", exist_ok=True)
    log = open(f"{a.out}/train_continue.jsonl", "a"); t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    for step in range(START + 1, a.total_steps + 1):
        ts = time.time(); si, _ = allB[step - 1]; n = len(si)
        gdiag = None
        if step in (51, 100, 200, 300, 400, 500):          # per-loss weighted grad norms on item 0 (separate backward, discarded)
            gdiag = {}
            for nm, w in (("cos", LAM_COS), ("exec", LAM_EXEC)):
                opt.zero_grad(set_to_none=True); _, parts, _ = item_loss(S, singles[si[0]]); (w * parts[nm]).backward()
                gdiag[nm + "_weighted"] = {"projector": grad_norm([p for _, p in S.proj]), "lora": grad_norm([p for _, p in S.lora])}
        opt.zero_grad(set_to_none=True); recs = []; ntok = 0
        for i in si:
            loss, parts, info = item_loss(S, singles[i]); (loss / n).backward()
            recs.append({"cos": float(parts["cos"]), "exec": float(parts["exec"]), **info}); ntok += len(S.target_ids(singles[i]["execution"]))
        gp, gl = grad_norm([p for _, p in S.proj]), grad_norm([p for _, p in S.lora])
        fin = all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0)); opt.step()
        m = lambda k: float(np.mean([r[k] for r in recs])); dt = time.time() - ts
        rec = {"step": step, "loss_cos_raw": m("cos"), "loss_exec_raw": m("exec"), "loss_cos_weighted": LAM_COS * m("cos"), "loss_exec_weighted": LAM_EXEC * m("exec"),
               "loss_total": LAM_COS * m("cos") + LAM_EXEC * m("exec"), "fr_cosd": m("fr_cosd"), "adj_cos": m("adj_cos"),
               "grad_total_preclip": gn, "grad_proj": gp, "grad_lora": gl, "grads_finite": fin, "clipped": gn > 1.0, "lr": opt.param_groups[0]["lr"],
               "grad_per_loss_item0": gdiag, "step_s": dt, "exec_tokens": ntok, "tok_per_s": ntok / dt, "elapsed_s": time.time() - t0,
               "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "batch": si, "epoch": (step - 1) * a.n_single / len(singles)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"step {step} cos_tf {rec['loss_cos_raw']:.4f} cos_fr {rec['fr_cosd']:.4f} exec {rec['loss_exec_raw']:.4f} total {rec['loss_total']:.4f} "
              f"gn {gn:.3f} (p {gp:.4f} l {gl:.4f}) fin {fin} {dt:.1f}s {rec['elapsed_s']:.0f}s mem {rec['peak_mem_gb']:.0f}G {gdiag or ''}", flush=True)
        if not fin or not np.isfinite(rec["loss_total"]):
            print("STOP: non-finite"); save_ck(S, opt, f"{a.out}/checkpoints/joint_step{step}_nonfinite.pt", step, {"phase": "joint"}); break
        if step in save_at:
            save_ck(S, opt, f"{a.out}/checkpoints/joint_step{step}.pt", step, {"phase": "joint", "lam_cos": LAM_COS, "lam_exec": LAM_EXEC, "lr": a.lr,
                                                                                "resumed_from": a.init, "continuation": "strict (weights+AdamW+RNG+data order)"})
        if step in diag_at:
            td = time.time(); r = diag(S, vals, a.n_val, f"in-process global step {step}"); r["diag_seconds"] = time.time() - td
            json.dump(r, open(f"{a.out}/evaluations/diag_joint_step{step}.json", "w"), indent=1)
            print(f"[diag {step}] FR paired {r['fr']['paired']:.4f} shuf {r['fr']['shuffled_perm']:.4f} gap {r['fr']['gap_perm']['mean']:+.4f} {r['fr']['gap_perm']['ci95']} "
                  f"crossq {r['collapse']['zs_crossq_cos']:.4f} var {r['collapse']['zs_var_over_teacher_var']:.4f}", flush=True)


if __name__ == "__main__":
    main()
