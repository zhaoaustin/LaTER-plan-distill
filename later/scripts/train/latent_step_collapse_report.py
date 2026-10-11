"""CPU-only report for diag_latent_step_collapse: plot, examples.md, metadata.json (reads saved outputs; no model)."""
import argparse
import hashlib
import json
import os
import re
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

TYPES = [("integral / calculus", r"\\int|integral|derivative|limit|\\lim"), ("probability", r"probab"),
         ("counting / combinatorics", r"how many ways|arrange|choose|combinations|permutation|committee"),
         ("number theory", r"remainder|divisible|prime|digit|gcd|modulo|\\pmod|integer[s]? n"),
         ("geometry", r"triangle|circle|angle|area|polygon|radius|square|rectangle"),
         ("trigonometry", r"\\sin|\\cos|\\tan|sin |cos "), ("sequence / series", r"sequence|series|\\sum|term"),
         ("polynomial / roots", r"polynomial|roots|root of|x\^3|x\^2"), ("word problem", r"\$\d|dollars|students|cost|price|hours|miles")]


def mtype(q):
    for name, pat in TYPES:
        if re.search(pat, q, flags=re.I):
            return name
    return "other algebra"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--repo", required=True)
    a = ap.parse_args()
    d = pd.read_csv(os.path.join(a.out, "per_step_metrics.csv"))
    js = json.load(open(os.path.join(a.out, "per_step_metrics.json")))
    sw = json.load(open(os.path.join(a.out, "step_swap_metrics.json")))
    lat = torch.load(os.path.join(a.out, "latents_fp16.pt"))
    H, Z = lat["h_in"].double().numpy(), lat["z_out"].double().numpy()
    N, L, _ = Z.shape
    # ---------------- plot
    g = lambda rep, col: d[d.representation == rep].sort_values("step")[col].values
    st = np.arange(1, L + 1)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for rep, c in (("h_in", "C0"), ("z_raw", "C2"), ("z_out", "C3")):
        ax[0, 0].plot(st, g(rep, "relative_cross_question_std"), "o-", c=c, label=rep)
    ax[0, 0].plot(st[1:], g("ref_explicit_plan_hidden", "relative_cross_question_std")[1:], "s--", c="C0", alpha=.4,
                  label="ref: explicit hidden @ gold plan tokens")
    ax[0, 0].plot(st[1:], g("ref_explicit_plan_token_embed", "relative_cross_question_std")[1:], "s--", c="C3", alpha=.4,
                  label="ref: gold plan token embeddings")
    ax[0, 0].set_yscale("log"); ax[0, 0].set_title("A2 relative cross-question std (centred / RMS)"); ax[0, 0].legend(fontsize=7)
    for rep, c in (("h_in", "C0"), ("z_out", "C3")):
        ax[0, 1].plot(st, g(rep, "cos_raw_mean"), "o-", c=c, label=f"{rep} mean")
        ax[0, 1].fill_between(st, g(rep, "cos_raw_p10"), g(rep, "cos_raw_p90"), color=c, alpha=.2, label=f"{rep} P10-P90")
    ax[0, 1].set_title("A1 cross-question raw cosine (i != j)"); ax[0, 1].legend(fontsize=7)
    for rep, c in (("h_in", "C0"), ("z_out", "C3")):
        ax[1, 0].plot(st, g(rep, "cka_with_step1"), "o-", c=c, label=f"{rep}: linear CKA with step 1")
        ax[1, 0].plot(st, g(rep, "nn_retrieval_vs_step1"), "x:", c=c, label=f"{rep}: top-1 identity retrieval vs step 1")
    ax[1, 0].plot(st, g("z_out", "cka_with_h_in_same_step"), "d-", c="C4", label="CKA(z_out_t, h_in_t)")
    ax[1, 0].set_title("question structure kept across steps / through projector"); ax[1, 0].legend(fontsize=7); ax[1, 0].set_ylim(0, 1.05)
    for rep, c in (("h_in", "C0"), ("z_out", "C3")):
        for k, ls in ((1, "-"), (2, "--")):
            v = [np.mean([np.dot(x[t], x[t + k]) / np.linalg.norm(x[t]) / np.linalg.norm(x[t + k]) for x in (H if rep == "h_in" else Z)])
                 for t in range(L - k)]
            ax[1, 1].plot(st[:L - k] + 0.5 * k, v, "o" + ls, c=c, label=f"{rep}: cos(t, t+{k}) same question")
    ax[1, 1].set_title("A3 within-question step similarity (x = midpoint)"); ax[1, 1].legend(fontsize=7)
    for x in ax.flat:
        x.set_xlabel("latent step t"); x.grid(alpha=.3)
    fig.suptitle("Stage-2 three-loss L=8, step 500 — 128 val questions (forward-only)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "latent_similarity_by_step.png"), dpi=130)
    # ---------------- examples
    val = pd.read_parquet(f"{a.data_dir}/stage2_val.parquet").iloc[:N]
    rid = lat["record_ids"]
    assert list(val.record_id) == rid
    types = [mtype(q) for q in val.question]
    pick, seen = [], set()
    for i, t in enumerate(types):
        if t not in seen:
            pick.append(i); seen.add(t)
    pick = pick[:12]
    mu = {k: X.mean(0) for k, X in (("h", H), ("z", Z))}
    def rel_dist(X, m, i, t):
        dd = np.linalg.norm(X[:, t] - m[t], axis=1)
        return dd[i] / dd.mean()
    def nn(X, i, t):
        C = X[:, t] - X[:, t].mean(0)
        C = C / np.linalg.norm(C, axis=1, keepdims=True)
        s = C @ C[i]; s[i] = -9
        return int(np.argmax(s)), float(s.max())
    pr = sw["_per_record"]
    with open(os.path.join(a.out, "examples.md"), "w") as f:
        f.write("# Test C — representative questions (three-loss L=8 step 500)\n\n"
                "Gold plan shown for review only; it is never fed to the student latent path. Per step t: L2 norm, "
                "relative distance to the cross-question mean (= ||x_i - mu_t|| / mean_j ||x_j - mu_t||; 1 = average), "
                "and the nearest other question (centred cosine). ΔCE = swapped − original, mean over non-repeated "
                "execution tokens (controlled forced forward, donor fixed).\n\n")
        for i in pick:
            r = val.iloc[i]
            f.write(f"## {rid[i]} — keyword type (rough): {types[i]}\n\n**Q:** {r.question[:350].strip()}\n\n**Gold plan (excerpt):** "
                    f"{r.plan[:350].strip()!r}\n\n| t | ‖h_in‖ | ‖z_out‖ | rel. dist h_in | rel. dist z_out | NN question (z_out, cos) |\n|---|---|---|---|---|---|\n")
            for t in (0, 1, 3, 7):
                j, s = nn(Z, i, t)
                f.write(f"| {t + 1} | {np.linalg.norm(H[i, t]):.1f} | {np.linalg.norm(Z[i, t]):.3f} | {rel_dist(H, mu['h'], i, t):.2f} | "
                        f"{rel_dist(Z, mu['z'], i, t):.2f} | {types[j]} ({s:.2f}) |\n")
            p = pr[rid[i]]
            o = p["original"]["ce_nonrep"]
            f.write("\nStep swap ΔCE: " + ", ".join(f"{c} {p[c]['ce_nonrep'] - o:+.5f}" for c in
                                                  ("swap_step_1", "swap_step_8", "swap_prefix_1:4", "swap_all_1:8")) + "\n\n")
        # pair curves
        f.write("## Pair curves: centred distance ||z_i - z_j|| / mean pairwise distance at that step\n\n| pair | " +
                " | ".join(f"t={t}" for t in range(1, L + 1)) + " |\n|---|" + "---|" * L + "\n")
        for i, j in ((pick[0], pick[1]), (pick[2], pick[3]), (pick[4], pick[5])):
            row = []
            for t in range(L):
                X = Z[:, t]
                sq = (X ** 2).sum(1)
                D = np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * X @ X.T, 0))
                row.append(D[i, j] / D[~np.eye(N, dtype=bool)].mean())
            f.write(f"| {types[i]} vs {types[j]} | " + " | ".join(f"{x:.2f}" for x in row) + " |\n")
        # swap cases: largest |delta| for swap_all and one median
        dl = sorted(((pr[k]["swap_all_1:8"]["ce_nonrep"] - pr[k]["original"]["ce_nonrep"], k) for k in pr))
        f.write("\n## Step-swap cases (swap_all_1:8; largest decrease, largest increase, median)\n\n")
        for dv, k in (dl[0], dl[-1], dl[len(dl) // 2]):
            i = rid.index(k)
            p = pr[k]
            f.write(f"- **{k}** ({types[i]}), donor {p['donor']}: original CE {p['original']['ce_nonrep']:.5f}, "
                    f"swap_all {p['swap_all_1:8']['ce_nonrep']:.5f} (Δ {dv:+.5f}), swap_step_1 Δ "
                    f"{p['swap_step_1']['ce_nonrep'] - p['original']['ce_nonrep']:+.5f}, first-token KL {p['swap_all_1:8']['kl0']:.2e}. "
                    f"Q: {val.iloc[i].question[:160].strip()!r}\n")
        allv = np.array([x for x, _ in dl])
        f.write(f"\nAll 128: swap_all ΔCE range [{allv.min():+.5f}, {allv.max():+.5f}], |Δ| median {np.median(np.abs(allv)):.5f}; "
                f"original per-question CE median {np.median([pr[k]['original']['ce_nonrep'] for k in pr]):.4f}.\n")
    # ---------------- metadata
    git = lambda *x: subprocess.run(["git", "-C", a.repo, *x], capture_output=True, text=True).stdout.strip()
    meta = {"model_checkpoint": a.model, "checkpoint_step": 500,
            "checkpoint_stage2_meta": json.load(open(os.path.join(a.model, "stage2_meta.json"))),
            "training_run_config": os.path.join(os.path.dirname(a.model), "run_config_resolved.json"),
            "training_git_commit": "a4f1753b43e968bb8dd4c2a64c6cef6ea1700cfb (wandb-metadata of run 804caa03a65b)",
            "diagnostic_repo_head": git("rev-parse", "HEAD"), "diagnostic_branch": git("branch", "--show-current"),
            "modeling_diff_vs_training_commit": "additive only (latent_override / PlanRec helpers); builder path unchanged",
            "latent_steps": L, "latent_projector_output_rms": 0.02195320464670658, "latent_projector_dropout": 0.0,
            "data": f"{a.data_dir}/stage2_val.parquet, first {N} records (= training-time probe set, probe_n=128)",
            "data_sha256": hashlib.sha256(open(f"{a.data_dir}/stage2_val.parquet", "rb").read()).hexdigest(),
            "record_ids": rid, "n_distinct_questions": js.get("n_distinct_questions"),
            "precision": "fp32 master weights, bf16 autocast (same as training probe)", "mode": "model.eval() + torch.inference_mode()",
            "gpu": "NVIDIA B200 (Slurm job 48907334, node e4-1)", "torch": "2.8.0+cu128", "transformers": "5.12.1",
            "examples_picked": [rid[i] for i in pick]}
    json.dump(meta, open(os.path.join(a.out, "metadata.json"), "w"), indent=1)
    print("report done", [rid[i] for i in pick])


if __name__ == "__main__":
    main()
