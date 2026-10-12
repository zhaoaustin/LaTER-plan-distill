"""Collect contrastive_2b outputs -> metrics.csv, eval.json, cases.jsonl and the four figures (fig1..fig4)."""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

O = sys.argv[1]
E = f"{O}/evaluations"
RUNS = ["D0", "A001", "A005", "A01", "B002", "B005"]
J = lambda p: json.load(open(p)) if os.path.exists(p) else None


def rows(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


# ---- metrics.csv
tr = {r: rows(f"{O}/train_{r}.jsonl") for r in RUNS}
keys = ["tag", "step_additional", "step_global", "lam_anchor", "lam_plan", "loss_cos_raw", "loss_exec_raw", "loss_anchor_raw", "jsd_aware_blind",
        "loss_plan_matched_raw", "loss_plan_rank_raw", "plan_dA", "plan_dB", "loss_cos_weighted", "loss_exec_weighted", "loss_anchor_weighted",
        "loss_plan_weighted", "fr_cosd", "grad_total_preclip", "grad_proj", "grad_lora", "grads_finite", "elapsed_s", "peak_mem_gb"]
gkeys = ["cos_weighted", "exec_weighted", "anchor_weighted", "plan_matched_weighted", "plan_rank_weighted"]
with open(f"{O}/metrics.csv", "w") as f:
    f.write(",".join(keys + [f"g_{k}_{w}" for k in gkeys for w in ("proj", "lora")]) + "\n")
    for r in RUNS:
        for x in tr[r]:
            g = x.get("grad_per_term_item0") or {}
            f.write(",".join("" if x.get(k) is None else str(x[k]) for k in keys) + "," +
                    ",".join(str(g[k]["projector" if w == "proj" else "lora"]) if k in g else "" for k in gkeys for w in ("proj", "lora")) + "\n")

# ---- eval.json
ck = ["start50"] + [f"{r}_add{s}" for r in RUNS for s in (10, 25, 50)]
ev = {}
for c in ck:
    d = {k: J(f"{E}/{k}_{c}.json") for k in ("probe", "extra", "acc", "summary")}
    if any(d.values()):
        if d["probe"]:
            d["probe"] = {k: v for k, v in d["probe"].items() if k != "per_sample"}
        ev[c] = d
json.dump(ev, open(f"{O}/eval.json", "w"), indent=1)

# ---- cases.jsonl (held-out 8 + choice-valid train pairs: generations under self / crossq / mean / zero / teacher_A / teacher_B)
with open(f"{O}/cases.jsonl", "w") as f:
    for c in ("start50", "D0_add50"):
        xr = {r["id"]: r for r in rows(f"{E}/extra_rows_{c}.jsonl") if r["kind"] == "pair"}
        pairs = {json.loads(l)["id"]: json.loads(l) for l in open("/scratch/hz33227/e2c/stage2/outputs/stage2b_executor_50step/data/pairs.jsonl")}
        for r in rows(f"{E}/rows_{c}.jsonl"):
            if r["kind"] != "pair" or "gens" not in r:
                continue
            p = pairs[r["id"]]; x = xr.get(r["id"][:11], {})
            f.write(json.dumps({"ckpt": c, "question_id": r["id"], "role": r["role"], "choice_valid": r["choice_valid"], "default_plan": r["default"],
                                "method_label_A": p.get("method_label_A"), "method_label_B": p.get("method_label_B"), "plan_A": p["plan_A_norm"], "plan_B": p["plan_B_norm"],
                                "answer": p["answer"], "matched_vs_swapped_mean_logp": {k: x.get(k) for k in ("full", "span", "dspan")},
                                "generations": [{k: g[k] for k in ("cond", "pred", "correct", "ended", "exec_close", "ov_A", "ov_B", "terms_A", "terms_B", "text")} for g in r["gens"]]}) + "\n")

# ---- fig1: loss / grad vs steps
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for r in RUNS:
    x = [t["step_additional"] for t in tr[r]]
    if not x:
        continue
    ax[0, 0].plot(x, [t["loss_cos_raw"] for t in tr[r]], label=r); ax[0, 1].plot(x, [t["loss_exec_raw"] for t in tr[r]], label=r)
    if tr[r][0].get("jsd_aware_blind") is not None:
        ax[0, 2].plot(x, [t["jsd_aware_blind"] for t in tr[r]], label=f"{r} JSD(aware,blind)")
    if tr[r][0].get("loss_plan_rank_raw") is not None:
        ax[0, 2].plot(x, [t["loss_plan_rank_raw"] - np.log(2) for t in tr[r]], "--", label=f"{r} L_rank - ln2")
    ax[1, 0].plot(x, [t["grad_proj"] for t in tr[r]], label=r); ax[1, 1].plot(x, [t["grad_lora"] for t in tr[r]], label=r)
for i, r in enumerate(RUNS):
    for t in tr[r]:
        g = t.get("grad_per_term_item0")
        if not g:
            continue
        for j, k in enumerate(gkeys):
            if k in g:
                ax[1, 2].scatter(t["step_additional"] + 0.3 * i, max(g[k]["projector"], 1e-7), marker="o", c=f"C{j}", s=14)
                ax[1, 2].scatter(t["step_additional"] + 0.3 * i, max(g[k]["lora"], 1e-7), marker="x", c=f"C{j}", s=14)
for j, k in enumerate(gkeys):
    ax[1, 2].scatter([], [], c=f"C{j}", label=k)
ax[1, 2].set_yscale("log"); ax[1, 2].set_title("weighted per-term grad norm (o=projector, x=LoRA; item 0)")
for a_, t_ in zip(ax.flat, ["TF 1-cos (raw)", "FR exec CE (raw)", "anchor JSD / plan rank-ln2 (raw)", "grad norm projector (step batch)", "grad norm LoRA (step batch)"]):
    a_.set_title(t_); a_.set_xlabel("additional step")
for a_ in ax.flat:
    a_.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{O}/fig1_loss_grad_vs_steps.png", dpi=110); plt.close()

# ---- fig2: latent collapse vs steps
fig, ax = plt.subplots(1, 4, figsize=(17, 4))
pts = {"D0": [("start50", 0), ("D0_add10", 10), ("D0_add25", 25), ("D0_add50", 50)]}
for r in RUNS[1:]:
    pts[r] = [("start50", 0), (f"{r}_add10", 10)]
for r, P in pts.items():
    P = [(c, s) for c, s in P if c in ev and ev[c]["probe"]]
    s = [x[1] for x in P]; pr = [ev[c]["probe"] for c, _ in P]
    ax[0].errorbar(s, [p["tf_gap"]["mean"] for p in pr], yerr=np.array([[p["tf_gap"]["mean"] - p["tf_gap"]["ci95"][0], p["tf_gap"]["ci95"][1] - p["tf_gap"]["mean"]] for p in pr]).T, marker="o", label=r, capsize=3)
    ax[1].errorbar(s, [p["fr_gap"]["mean"] for p in pr], yerr=np.array([[p["fr_gap"]["mean"] - p["fr_gap"]["ci95"][0], p["fr_gap"]["ci95"][1] - p["fr_gap"]["mean"]] for p in pr]).T, marker="o", label=r, capsize=3)
    ax[2].plot(s, [p["zs_var_over_teacher_var"] for p in pr], marker="o", label=r); ax[3].plot(s, [p["zs_crossq_cos"] for p in pr], marker="o", label=r)
for a_, t_ in zip(ax, ["TF paired - shuffled cos (48 unseen, 95% CI)", "FR paired - shuffled cos", "var(Z_S)/var(Z_T) across questions", "Z_S cross-question cos"]):
    a_.set_title(t_, fontsize=9); a_.set_xlabel("additional step"); a_.legend(fontsize=7)
ax[0].axhline(0, c="k", lw=0.5); ax[1].axhline(0, c="k", lw=0.5)
plt.tight_layout(); plt.savefig(f"{O}/fig2_latent_collapse_vs_steps.png", dpi=110); plt.close()

# ---- fig3: bypass sensitivity vs accuracy / gold CE
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for c, d in ev.items():
    if not d["extra"]:
        continue
    v = d["extra"]["val48"]
    ax[0].scatter(v["jsd_aware_blind_T"]["mean"], v["gold_ce_self"]); ax[0].annotate(c, (v["jsd_aware_blind_T"]["mean"], v["gold_ce_self"]), fontsize=7)
    ax[0].scatter(v["jsd_self_crossq_T"]["mean"], v["gold_ce_self"], marker="x", c="gray")
    if d["acc"]:
        ax[1].scatter(abs(v["dCE_crossq"]["mean"]) + abs(v["dCE_zero"]["mean"]), d["acc"]["accuracy"])
        ax[1].annotate(f"{c} ({d['acc']['n_correct']}/{d['acc']['n']})", (abs(v["dCE_crossq"]["mean"]) + abs(v["dCE_zero"]["mean"]), d["acc"]["accuracy"]), fontsize=7)
ax[0].set_xlabel("JSD aware|blind on T (o) / JSD self|crossq (x), 48 unseen"); ax[0].set_ylabel("gold exec CE (self latent)")
ax[1].set_xlabel("|dCE zero| + |dCE crossq| (48 unseen)"); ax[1].set_ylabel("accuracy, 128 independent unseen questions")
ax[0].set_xscale("log"); plt.tight_layout(); plt.savefig(f"{O}/fig3_bypass_vs_accuracy.png", dpi=110); plt.close()

# ---- fig4: plan-following 2x2 (likelihood, held-out) + generated-strategy confusion
cks = [c for c in ("start50", "D0_add50", "B002_add10", "B005_add10") if c in ev and ev[c]["extra"]]
fig, ax = plt.subplots(2, len(cks), figsize=(4 * len(cks), 7.5), squeeze=False)
for i, c in enumerate(cks):
    m = ev[c]["extra"]["heldout8"]["span_2x2_mean"]
    M = np.array([[m["EA_ZA"], m["EA_ZB"]], [m["EB_ZA"], m["EB_ZB"]]])
    ax[0, i].imshow(M - M.mean(1, keepdims=True), cmap="coolwarm", vmin=-0.01, vmax=0.01)
    for (a_, b_), val in np.ndenumerate(M):
        ax[0, i].text(b_, a_, f"{val:.4f}", ha="center", va="center", fontsize=8)
    ax[0, i].set_xticks([0, 1], ["Z_T^A", "Z_T^B"]); ax[0, i].set_yticks([0, 1], ["E_A", "E_B"]); ax[0, i].set_title(f"{c}\nheld-out mean log p (row-centered colour)", fontsize=8)
    s = ev[c].get("summary")
    if s:
        G = [r for r in rows(f"{E}/rows_{c}.jsonl") if r["kind"] == "pair" and r["role"] == "heldout8" and "gens" in r]
        lab = lambda x: 0 if x["ov_A"] > x["ov_B"] else 1
        CM = np.zeros((2, 2))
        for r in G:
            for zi, cond in enumerate(("teacher_A", "teacher_B")):
                CM[zi, lab(next(x for x in r["gens"] if x["cond"] == cond))] += 1
        ax[1, i].imshow(CM, cmap="Blues", vmin=0, vmax=8)
        for (a_, b_), val in np.ndenumerate(CM):
            ax[1, i].text(b_, a_, int(val), ha="center", va="center")
        ax[1, i].set_xticks([0, 1], ["gen ~ E_A", "gen ~ E_B"]); ax[1, i].set_yticks([0, 1], ["inject Z_T^A", "inject Z_T^B"])
        ax[1, i].set_title("held-out 8: generated strategy (auto n-gram label)", fontsize=8)
    else:
        ax[1, i].axis("off"); ax[1, i].text(0.1, 0.5, "no generation eval\n(10-step diagnostic ckpt)", fontsize=8)
plt.tight_layout(); plt.savefig(f"{O}/fig4_plan_following_2x2.png", dpi=110); plt.close()
print("ok", list(ev))
