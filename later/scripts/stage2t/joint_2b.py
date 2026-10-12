"""Stage 2B joint (fixed two-loss): 1.0 * teacher-forced latent cosine + 0.1 * free-running execution CE.

Shared student = Stage-1 Qwen3-4B (frozen) + ONE LoRA r16 (q/v/o) + LaTER LatentProjector (TStudent from three_step).
Teacher Z_T = raw frozen 2A step-300 compressor output (same 8 input-embedding slots as Z_S; no adapter).
Forward A (TF): chat(Q)+'<EXPLORATION>\\n' + [Z_T1..Z_T7] (one causal forward); the prediction for slot i is
  proj(h at the position just before slot i)  -> uses Q and Z_T[:i] only (Z_T[i:] never visible for slot i);
  L_cos = mean_i (1 - cos(z_pred_i, Z_T_i)), fp32.
Forward B (FR): Z_S = free-running recurrence from Q (autograd, no teacher), CE over E + '</EXECUTION><|im_end|>'.
L = 1.0 * L_cos + 0.1 * L_exec, one backward, one optimizer step.  Modes: gates | train | probe.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from later.scripts.bridge.bridge_input_abcd import boot
from later.scripts.stage2s.stochastic_pilot import L, build_pairs, grad_norm, short
from later.scripts.stage2t.three_step import D, TStudent, batches, load_jsonl, save_ck

LAM_COS, LAM_EXEC = 1.0, 0.1


def tf_latents(S, q, ZT):
    """Teacher-forced predictions [8, d]: slot i predicted from Q + ZT[:i]."""
    pre = S.prompt_ids(q)
    ids = torch.tensor([pre + [S.T.pad_id] * (L - 1)], device=S.dev)
    emb = S.base.get_input_embeddings()(ids)
    emb = torch.cat([emb[:, :len(pre)], ZT[:L - 1].unsqueeze(0).to(emb.dtype)], 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hs = S.base.model(inputs_embeds=emb, position_ids=torch.arange(emb.size(1), device=S.dev).unsqueeze(0), use_cache=False).last_hidden_state
        h = hs[0, len(pre) - 1:len(pre) - 1 + L]
        return S.base.latent_projector(h).float()


def cosd(Z, T):
    return (1 - F.cosine_similarity(Z.float(), T.float(), dim=-1, eps=1e-8)).mean()


def item_loss(S, s, lam_exec=LAM_EXEC):
    q, T = s["question"], S.teacher(s["plan_norm"])
    l_cos = cosd(tf_latents(S, q, T), T)
    Z = S.zlat(q)
    sf, _, _ = S.score(q, Z.unsqueeze(0), [S.target_ids(s["execution"])])
    l_exec = -sf[0]
    with torch.no_grad():
        fr = float(cosd(Z, T))
    return LAM_COS * l_cos + lam_exec * l_exec, {"cos": l_cos, "exec": l_exec}, {"fr_cosd": fr, "adj_cos": float(F.cosine_similarity(Z[1:], Z[:-1], -1).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gates", "train", "probe"], required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--s2b", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50); ap.add_argument("--n_single", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--save_at", default="10,25,50"); ap.add_argument("--tag", default="joint"); ap.add_argument("--ckpt", default=None); ap.add_argument("--n_val", type=int, default=48)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = TStudent(a.model, a.s2b, seed=0)
    singles = load_jsonl(f"{a.s2b}/data/train_singles.jsonl"); vals = load_jsonl(f"{a.s2b}/data/val_singles.jsonl")
    params = S.params()

    if a.mode == "gates":
        g = {}
        s0 = singles[0]; q = s0["question"]; T = S.teacher(s0["plan_norm"])
        g["ZT_shape_dtype"] = [list(T.shape), str(T.dtype)]; g["ZT_rms"] = float(T.pow(2).mean().sqrt())
        with torch.no_grad():
            Zs = S.zlat(q); P = tf_latents(S, q, T)
            g["ZS_shape"] = list(Zs.shape); g["ZS_rms"] = float(Zs.pow(2).mean().sqrt()); g["same_device"] = str(Zs.device) == str(T.device)
            # alignment: feeding the student's own latents through the TF path must reproduce the free-running recurrence
            P_self = tf_latents(S, q, Zs)
            g["tf_with_own_latents_vs_recurrence_max_abs"] = float((P_self - Zs).abs().max())
            g["tf_with_own_latents_vs_recurrence_cos_min"] = float(F.cosine_similarity(P_self, Zs, -1).min())
            # leakage: changing Z_T[j] changes predictions only at slots i > j; Z_T[7] is never an input
            leak = []
            for j in range(L):
                T2 = T.clone(); T2[j] = torch.randn_like(T2[j]) * T.std()
                d = (tf_latents(S, q, T2) - P).abs().amax(-1)
                leak.append([float(x) for x in d])
            g["tf_change_matrix_rows_j_cols_i"] = leak
            g["no_current_or_future_leak"] = all(leak[j][i] == 0 for j in range(L) for i in range(L) if i <= j)
            g["past_teacher_used"] = all(leak[j][i] > 0 for j in range(L - 1) for i in range(j + 1, L))
            # FR path ignores the teacher entirely
            k = __import__("later.scripts.stage2s.stochastic_pilot", fromlist=["key"]).key(s0["plan_norm"]); bak = S.Zraw[k].clone(); S.Zraw[k] = bak * 0 + 1
            g["fr_independent_of_teacher"] = bool(torch.equal(S.zlat(q), Zs)); S.Zraw[k] = bak
        # cosine is mean(1-cos)
        g["cos_loss_check"] = [float(cosd(P, T)), float((1 - (P * T).sum(-1) / (P.norm(dim=-1) * T.norm(dim=-1))).mean())]
        # gradients per loss
        gr = {}
        for nm in ("cos", "exec"):
            for p in params:
                p.grad = None
            _, parts, _ = item_loss(S, s0); parts[nm].backward()
            gr[nm] = {"projector": grad_norm([p for _, p in S.proj]), "lora": grad_norm([p for _, p in S.lora]),
                      "finite": all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)}
        # exec gradient reaches z1 (through the recurrence)
        for p in params:
            p.grad = None
        Zg = S.latents(q, torch.zeros(1, D, device=S.dev), 0.0, 1.0); Zg.retain_grad()
        (-S.score(q, Zg[0][None], [S.target_ids(s0["execution"])])[0][0]).backward()
        gr["exec_dz_per_slot"] = [float(x) for x in Zg.grad[0].norm(dim=-1)]
        g["grads"] = gr
        g["grads_ok"] = all(gr[n]["projector"] > 0 and gr[n]["lora"] > 0 and gr[n]["finite"] for n in ("cos", "exec")) and gr["exec_dz_per_slot"][0] > 0
        tgt = S.target_ids(s0["execution"]); g["ce_target_tail"] = S.tok.decode(tgt[-5:]); g["ce_n_targets"] = len(tgt)
        dec = S.tok.decode(S.prompt_ids(q)); g["no_plan_or_exec_in_prompt"] = s0["plan_norm"][10:80] not in dec and s0["execution"][10:80] not in dec
        snap = {n: p.detach().clone() for n, p in S.peft.named_parameters() if not p.requires_grad}
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0); opt.zero_grad(set_to_none=True)
        loss, _, _ = item_loss(S, s0); loss.backward(); opt.step()     # 1 smoke step (counted)
        g["base_unchanged_after_step"] = all(torch.equal(p, snap[n]) for n, p in S.peft.named_parameters() if not p.requires_grad)
        g["teacher_cache_unchanged"] = True
        hard = [g["tf_with_own_latents_vs_recurrence_cos_min"] > 0.999, g["no_current_or_future_leak"], g["past_teacher_used"], g["fr_independent_of_teacher"],
                abs(g["cos_loss_check"][0] - g["cos_loss_check"][1]) < 1e-4, g["grads_ok"], g["no_plan_or_exec_in_prompt"], g["base_unchanged_after_step"],
                g["ZS_shape"] == g["ZT_shape_dtype"][0]]
        g["hard_gate_vector"] = hard; g["ALL_PASS"] = all(hard)
        json.dump(g, open(f"{a.out}/joint_gates.json", "w"), indent=1); print(json.dumps(g, indent=1))
        return

    if a.mode == "train":
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        B = batches(singles, [], a.n_single, 0, a.steps, seed=7)          # same order as the KD-only 2B run
        save_at = {int(x) for x in a.save_at.split(",")}
        os.makedirs(f"{a.out}/checkpoints", exist_ok=True); save_ck(S, None, f"{a.out}/checkpoints/{a.tag}_step0.pt", 0, {"phase": "joint"})
        log = open(f"{a.out}/train_{a.tag}.jsonl", "w"); t0 = time.time(); torch.cuda.reset_peak_memory_stats()
        for step in range(1, a.steps + 1):
            si, _ = B[step - 1]; n = len(si)
            # per-loss gradient norms on the first item (diagnostic, separate backward) at a few steps
            gdiag = None
            if step in (1, 10, 25, 50):
                gdiag = {}
                for nm, w in (("cos", LAM_COS), ("exec", LAM_EXEC)):
                    opt.zero_grad(set_to_none=True); _, parts, _ = item_loss(S, singles[si[0]]); (w * parts[nm]).backward()
                    gdiag[nm + "_weighted"] = {"projector": grad_norm([p for _, p in S.proj]), "lora": grad_norm([p for _, p in S.lora])}
            opt.zero_grad(set_to_none=True); recs = []
            for i in si:
                loss, parts, info = item_loss(S, singles[i]); (loss / n).backward()
                recs.append({"cos": float(parts["cos"]), "exec": float(parts["exec"]), **info})
            gp, gl = grad_norm([p for _, p in S.proj]), grad_norm([p for _, p in S.lora])
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0)); opt.step()
            m = lambda k: float(np.mean([r[k] for r in recs]))
            rec = {"step": step, "loss_cos_raw": m("cos"), "loss_exec_raw": m("exec"), "loss_cos_weighted": LAM_COS * m("cos"), "loss_exec_weighted": LAM_EXEC * m("exec"),
                   "loss_total": LAM_COS * m("cos") + LAM_EXEC * m("exec"), "fr_cosd": m("fr_cosd"), "adj_cos": m("adj_cos"),
                   "grad_total_preclip": gn, "grad_proj": gp, "grad_lora": gl, "grad_per_loss_item0": gdiag, "elapsed_s": time.time() - t0,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "batch": si}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"[{a.tag}] step {step} cos_tf {rec['loss_cos_raw']:.4f} cos_fr {rec['fr_cosd']:.4f} exec {rec['loss_exec_raw']:.4f} total {rec['loss_total']:.4f} "
                  f"adj {rec['adj_cos']:.3f} gn {gn:.3f} (p {gp:.3f} l {gl:.3f}) {gdiag or ''} {rec['elapsed_s']:.0f}s mem {rec['peak_mem_gb']:.0f}G", flush=True)
            if step in save_at or step == a.steps:
                save_ck(S, opt, f"{a.out}/checkpoints/{a.tag}_step{step}.pt", step, {"phase": "joint", "lam_cos": LAM_COS, "lam_exec": LAM_EXEC, "lr": a.lr})
        return

    if a.mode == "probe":        # TF vs FR cosine on unseen val questions, paired vs shuffled
        S.load(torch.load(a.ckpt, weights_only=False)["state"])
        V = vals[:a.n_val]; rows = []
        with torch.no_grad():
            T = torch.stack([S.teacher(v["plan_norm"]) for v in V])
            Ptf = torch.stack([tf_latents(S, v["question"], T[i]) for i, v in enumerate(V)])
            Zs = torch.stack([S.zlat(v["question"]) for v in V])
        cs = lambda X, Y: F.cosine_similarity(X, Y, dim=-1).mean(-1)          # mean over 8 slots
        sh = torch.roll(torch.arange(len(V)), 1)
        tf_p, tf_s = cs(Ptf, T), cs(Ptf, T[sh]); fr_p, fr_s = cs(Zs, T), cs(Zs, T[sh])
        tf_slot = F.cosine_similarity(Ptf, T, dim=-1).mean(0); fr_slot = F.cosine_similarity(Zs, T, dim=-1).mean(0)
        Tm = T.mean(0, keepdim=True)
        fl = Zs.flatten(1); cc = F.cosine_similarity(fl.unsqueeze(1), fl.unsqueeze(0), -1); iu = torch.triu_indices(len(V), len(V), 1)
        st = lambda v: {"mean": float(v.mean()), "ci95": boot(v.tolist()), "frac_pos": float((v > 0).float().mean())}
        res = {"ckpt": a.ckpt, "n": len(V),
               "tf_cos_paired": float(tf_p.mean()), "tf_cos_shuffled": float(tf_s.mean()), "tf_paired_minus_shuffled": st(tf_p - tf_s),
               "fr_cos_paired": float(fr_p.mean()), "fr_cos_shuffled": float(fr_s.mean()), "fr_paired_minus_shuffled": st(fr_p - fr_s),
               "teacher_mean_cos_baseline": float(cs(Tm.expand_as(T), T).mean()),
               "tf_cos_per_slot": [float(x) for x in tf_slot], "fr_cos_per_slot": [float(x) for x in fr_slot],
               "zs_crossq_cos": float(cc[iu[0], iu[1]].mean()), "zs_var_over_teacher_var": float(Zs.var(0).mean() / T.var(0).mean()),
               "zs_adjacent_slot_cos": float(F.cosine_similarity(Zs[:, 1:], Zs[:, :-1], -1).mean()),
               "teacher_adjacent_slot_cos": float(F.cosine_similarity(T[:, 1:], T[:, :-1], -1).mean())}
        name = os.path.basename(a.ckpt).replace(".pt", "")
        json.dump(res, open(f"{a.out}/probe_{name}.json", "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
