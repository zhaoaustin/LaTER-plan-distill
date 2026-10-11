"""P0 residual amplification (forward-only: model.eval() + torch.inference_mode(), no optimizer, no backward).

Offline latent intervention on an existing Stage-2 checkpoint: z_t^(a) = mu_t + a * (z_t - mu_t), mu_t = per-step mean of
the learned latents (projector output after output_rms = the embedding written into slot t) over a DISJOINT
calibration subset. Injected with latent_override: the 8 slot vectors are written into the student sequence and ONE
fresh full forward (use_cache=False, no KV cache of the learned recursion) computes the execution CE, so attention
states are consistent with the injected embeddings. alpha>1 latents are NOT produced by the model's own recursion.
correct_i: question i's transformed latents; shuffled_i: the transformed latents of donor perm[i] (fixed derangement,
seed 0, over the eval set), same slots / question / targets / masks.
rms_matched: each injected vector rescaled to the RMS of question i's original z_{i,t}.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from later.scripts.train.diag_latent_step_collapse import builder, exec_forward
from later.src.train.stage2_common import load_stage2_model, load_tokenizer
from later.src.train.stage2_plan_dataset import Stage2Collator, build_views, stage2_tokens
from later.src.train.stage2_planrec_monitor import fixed_derangement

ALPHAS = [1.0, 2.0, 4.0]


def boot(d, n=1000, seed=0):
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bs = np.sort(rng.choice(d, (n, len(d))).mean(1))
    return {"estimate": float(d.mean()), "ci95": [float(bs[int(0.025 * n)]), float(bs[int(0.975 * n) - 1])],
            "frac_positive": float((d > 0).mean()), "n": len(d)}


def zstats(Z, Z1):
    """Z, Z1 [N, L, d]: intervention vs alpha=1 latents."""
    rms = Z.pow(2).mean(-1).sqrt()
    C = Z - Z.mean(0, keepdim=True)
    Xn = torch.nn.functional.normalize(Z, dim=-1)
    N = Z.size(0)
    off = ~torch.eye(N, dtype=torch.bool, device=Z.device)
    cos = torch.stack([(Xn[:, t] @ Xn[:, t].T)[off].mean() for t in range(Z.size(1))])
    return {"latent_rms_mean": float(rms.mean()), "latent_rms_ratio_vs_alpha1": float(rms.mean() / Z1.pow(2).mean(-1).sqrt().mean()),
            "mean_l2_norm": float(Z.norm(dim=-1).mean()), "max_l2_norm": float(Z.norm(dim=-1).max()),
            "centered_rms": float(C.pow(2).mean().sqrt()),
            "centered_rms_ratio_vs_alpha1": float(C.pow(2).mean().sqrt() / (Z1 - Z1.mean(0, keepdim=True)).pow(2).mean().sqrt()),
            "cos_across_questions_per_step": [round(float(x), 6) for x in cos], "cos_across_questions_mean": float(cos.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_eval", type=int, default=128)
    ap.add_argument("--n_cal", type=int, default=128)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", required=True)
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
    tok = load_tokenizer(a.model)
    T = stage2_tokens(tok)
    vpath = f"{a.data_dir}/stage2_val.parquet"
    val = pd.read_parquet(vpath)
    ev = [val.iloc[i].to_dict() for i in range(a.n_eval)]                      # same 128 as earlier probes
    cal = [val.iloc[i].to_dict() for i in range(a.n_eval, a.n_eval + a.n_cal)]  # disjoint calibration subset
    assert not ({r["question"] for r in ev} & {r["question"] for r in cal})
    meta = json.load(open(os.path.join(a.model, "stage2_meta.json")))
    L = int(meta["latent_steps"])
    model = load_stage2_model(a.model, latent_embed_builder_no_grad=True, latent_projector_dropout=0.0,
                              dtype=torch.float32).cuda().eval()
    col = Stage2Collator(tok.pad_token_id)
    mk = lambda rows: [col([build_views(tok, r, L, T, 4096) for r in rows[s: s + a.bs]]) for s in range(0, len(rows), a.bs)]
    eb, cb = mk(ev), mk(cal)
    pre = {"checkpoint": a.model, "stage2_meta": meta, "latent_steps": L,
           "output_rms": float(model.latent_projector.output_rms), "training": model.training,
           "dropout_p": sorted({m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)}),
           "val_path": vpath, "val_sha256": hashlib.sha256(open(vpath, "rb").read()).hexdigest(),
           "eval_record_ids": [r["record_id"] for r in ev], "cal_record_ids": [r["record_id"] for r in cal],
           "eval_cal_question_overlap": 0}
    with torch.inference_mode():
        Zc = torch.cat([builder(model, b)[0] for b in cb])                      # [Nc, L, d] calibration latents
        mu = Zc.mean(0)                                                          # [L, d] per-step mean
        torch.save({"mu": mu.cpu(), "cal_record_ids": pre["cal_record_ids"], "definition":
                    "mu_t = mean over calibration questions of z_{i,t} (projector output after output_rms, slot input)"},
                   os.path.join(a.out, "latent_mu.pt"))
        Ze = torch.cat([builder(model, b)[0] for b in eb])                      # [Ne, L, d] eval learned latents
        # identity checks on the first batch: builder path vs override(Z) vs override(alpha=1 transform)
        b0 = eb[0]
        B0 = b0["student_input_ids"].size(0)
        ref = exec_forward(model, b0)
        ov = exec_forward(model, b0, latent_override=Ze[:B0])
        z_a1 = mu + 1.0 * (Ze[:B0] - mu)
        ov1 = exec_forward(model, b0, latent_override=z_a1)
        pre["identity"] = {
            "override_learned_vs_builder_max_abs_logit": float((ov["target_logits"] - ref["target_logits"]).abs().max()),
            "alpha1_transform_vs_learned_max_abs_latent": float((z_a1 - Ze[:B0]).abs().max()),
            "alpha1_transform_vs_builder_max_abs_logit": float((ov1["target_logits"] - ref["target_logits"]).abs().max()),
            "alpha1_transform_vs_builder_ce_diff": float(ov1["token_nll"].mean() - ref["token_nll"].mean()),
            "override_path": "forward_plan_distill_stage2(latent_override=...): builder not run, one full forward with "
                             "use_cache=False (no reused KV cache)"}
        pre["mu_vs_eval_mean_rel_diff_per_step"] = [round(float((mu[t] - Ze[:, t].mean(0)).norm() /
                                                           (Ze[:, t] - Ze[:, t].mean(0)).norm(dim=-1).mean()), 4) for t in range(L)]
        json.dump(pre, open(os.path.join(a.out, "preflight.json"), "w"), indent=1)
        P("PREFLIGHT", json.dumps(pre["identity"]), "mu-eval mean diff / mean residual", pre["mu_vs_eval_mean_rel_diff_per_step"])
        if pre["identity"]["alpha1_transform_vs_builder_ce_diff"] != 0.0 and abs(pre["identity"]["alpha1_transform_vs_builder_ce_diff"]) > 1e-4:
            raise SystemExit("alpha=1 identity failed")
        perm = fixed_derangement(a.n_eval, seed=0)
        rms_orig = Ze.pow(2).mean(-1, keepdim=True).sqrt()                      # [N, L, 1]
        variants, stats = {}, {}
        for al in ALPHAS:
            Za = mu + al * (Ze - mu)
            variants[f"alpha{al:g}_raw"] = Za
            stats[f"alpha{al:g}_raw"] = zstats(Za, Ze)
            if al != 1.0:
                Zm = Za * (rms_orig / Za.pow(2).mean(-1, keepdim=True).sqrt())
                variants[f"alpha{al:g}_rms_matched"] = Zm
                stats[f"alpha{al:g}_rms_matched"] = zstats(Zm, Ze)
        P("LATENT STATS", json.dumps({k: {kk: v[kk] for kk in ("latent_rms_ratio_vs_alpha1", "centered_rms_ratio_vs_alpha1",
                                                                  "cos_across_questions_mean", "max_l2_norm")} for k, v in stats.items()}))
        res = {}
        rows_out = []
        for name, Zv in variants.items():
            per = {"correct": {"ce": [], "ce16": []}, "shuffled": {"ce": [], "ce16": []}}
            kl16, ntok = [], []
            for bi, b in enumerate(eb):
                B = b["student_input_ids"].size(0)
                idx = list(range(bi * a.bs, bi * a.bs + B))
                outs = {"correct": exec_forward(model, b, latent_override=Zv[idx]),
                        "shuffled": exec_forward(model, b, latent_override=Zv[[perm[i] for i in idx]])}
                cnt = outs["correct"]["target_mask"].sum(1).tolist()
                lp = {k: torch.log_softmax(o["target_logits"].float(), -1) for k, o in outs.items()}
                o_ = 0
                for i in range(B):
                    sl = slice(o_, o_ + cnt[i])
                    for k, o in outs.items():
                        nll = o["token_nll"][sl].double()
                        per[k]["ce"].append(float(nll.mean()))
                        per[k]["ce16"].append(float(nll[:16].mean()))
                    pc, ps = lp["correct"][o_: o_ + min(16, cnt[i])], lp["shuffled"][o_: o_ + min(16, cnt[i])]
                    kl16.append(float((pc.exp() * (pc - ps)).sum(-1).mean()))
                    ntok.append(cnt[i])
                    o_ += cnt[i]
            g = np.array(per["shuffled"]["ce"]) - np.array(per["correct"]["ce"])
            g16 = np.array(per["shuffled"]["ce16"]) - np.array(per["correct"]["ce16"])
            w = np.array(ntok, dtype=np.float64)
            res[name] = {"exec_ce_correct": float(np.mean(per["correct"]["ce"])), "exec_ce_shuffled": float(np.mean(per["shuffled"]["ce"])),
                         "exec_ce_correct_token_weighted": float((np.array(per["correct"]["ce"]) * w).sum() / w.sum()),
                         "exec_ce_shuffled_token_weighted": float((np.array(per["shuffled"]["ce"]) * w).sum() / w.sum()),
                         "gain_shuffled": boot(g), "ce_first16_correct": float(np.mean(per["correct"]["ce16"])),
                         "ce_first16_shuffled": float(np.mean(per["shuffled"]["ce16"])), "gain_shuffled_first16": boot(g16),
                         "kl_first16_correct_vs_shuffled": {"mean": float(np.mean(kl16)), "median": float(np.median(kl16)),
                                                            "max": float(np.max(kl16))},
                         "latent_stats": stats[name]}
            for i in range(a.n_eval):
                rows_out.append({"variant": name, "record_id": ev[i]["record_id"], "donor_record_id": ev[perm[i]]["record_id"],
                                 "n_exec_tokens": ntok[i], "ce_correct": per["correct"]["ce"][i], "ce_shuffled": per["shuffled"]["ce"][i],
                                 "gain": g[i], "ce16_correct": per["correct"]["ce16"][i], "ce16_shuffled": per["shuffled"]["ce16"][i],
                                 "gain16": g16[i], "kl16_correct_vs_shuffled": kl16[i]})
            P(name, "ce_correct %.6f ce_shuffled %.6f gain %s kl16 %.3e" % (res[name]["exec_ce_correct"], res[name]["exec_ce_shuffled"],
                                                                        res[name]["gain_shuffled"], res[name]["kl_first16_correct_vs_shuffled"]["mean"]))
    # cross-variant: does the alpha=4 latent change predictions vs alpha=1 (correct)? recorded via ce differences
    res["_protocol"] = {"ce": "per-question mean NLL over all execution-side targets of the student view (E + A + closing "
                              "tags + EOS), teacher forced; summaries are means over questions; token-weighted also given",
                        "first16": "first 16 supervised execution targets", "kl": "KL(correct || shuffled) per position, "
                        "mean over the first 16 targets, then over questions", "bootstrap": "1000 resamples of questions, paired",
                        "shuffled": "fixed derangement seed 0 over the eval set; donor latents go through the same transform",
                        "intervention": "offline: z^(a) = mu_t + a (z - mu_t), not produced by the model's recursion"}
    json.dump(res, open(os.path.join(a.out, "p0_metrics.json"), "w"), indent=1)
    pd.DataFrame(rows_out).to_csv(os.path.join(a.out, "per_example.csv"), index=False)
    P("DONE in", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
