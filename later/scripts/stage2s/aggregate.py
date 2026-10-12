"""Aggregate stochastic-pilot eval shards (CPU): comparisons.csv, heldout_predictions.jsonl, summary json, generations md."""
import glob
import json
import sys

import numpy as np

from later.scripts.bridge.bridge_input_abcd import boot

O, tag = sys.argv[1], sys.argv[2]          # tag e.g. "25_tiny" or "0"
rows = [json.loads(l) for f in sorted(glob.glob(f"{O}/evaluations/eval_step{tag}_shard*.jsonl")) for l in open(f)]
K = 8
summ = {}
csv = ["split,subset,metric,value"]
for split in ("heldout8", "train_pair"):
    for subset in ("all", "choice_valid"):
        R = [r for r in rows if r["role"] == split and (subset == "all" or r["choice_valid"])]
        if not R:
            continue
        s = {"n_questions": len(R)}
        s["matched_both_pos"] = sum(r["matched_delta_A"] > 0 and r["matched_delta_B"] > 0 for r in R)
        per_q = [(r["matched_delta_A"] + r["matched_delta_B"]) / 2 for r in R]
        s["matched_delta_mean"] = float(np.mean(per_q)); s["matched_delta_ci95_by_q"] = boot(per_q)
        # teacher-free: does any eps flip the A/B preference sign within a question?
        s["pref_sign_diverse_questions"] = sum(len({x > 0 for x in r["pref_A_minus_B"][:K]}) == 2 for r in R)
        s["pref_std_over_eps_mean"] = float(np.mean([np.std(r["pref_A_minus_B"][:K]) for r in R]))
        idx = {c: r_i for r_i, c in enumerate(R[0]["conds"])}
        for c in ("self_sigma0", "zero", "mean", "crossq", "teacher_A", "teacher_B"):
            s[f"pref_abs_diff_self_vs_{c}"] = float(np.mean([abs(np.mean(r["pref_A_minus_B"][:K]) - r["pref_A_minus_B"][idx[c]]) for r in R]))
        # teacher diagnostic: does T_A vs T_B shift preference in the right direction?
        s["teacher_pref_TA_minus_TB_mean"] = float(np.mean([r["pref_A_minus_B"][idx["teacher_A"]] - r["pref_A_minus_B"][idx["teacher_B"]] for r in R]))
        s["teacher_pref_correct_sign"] = sum(r["pref_A_minus_B"][idx["teacher_A"]] > r["pref_A_minus_B"][idx["teacher_B"]] for r in R)
        for c in ("self_matched", "self_swapped", "self_sigma0", "zero", "mean", "crossq", "teacher"):
            s[f"exec_ce_{c}"] = float(np.mean([np.mean(r["exec_ce"][c]) for r in R]))
        for k in ("pairwise_cos_mean", "pairwise_dist_mean", "dist_to_sigma0_mean", "teacher_d_AB"):
            s[f"div_{k}"] = float(np.mean([r["diversity"][k] for r in R]))
        s["nearest_teacher_both_present"] = sum(len(set(r["nearest_teacher_per_eps"])) == 2 for r in R)
        if "gens" in R[0]:
            G = lambda r, c: next(g for g in r["gens"] if g["cond"] == c)
            s["gen_distinct_texts_over_8eps_mean"] = float(np.mean([len({G(r, f"self_eps{k}")["text"] for k in range(K)}) for r in R]))
            s["gen_eps_texts_all_identical_questions"] = sum(len({G(r, f"self_eps{k}")["text"] for k in range(K)}) == 1 for r in R)
            s["gen_eps_identical_to_sigma0"] = sum(all(G(r, f"self_eps{k}")["text"] == G(r, "self_sigma0")["text"] for k in range(K)) for r in R)
            for c in ["self_eps0", "self_sigma0", "zero", "mean", "crossq", "teacher_A", "teacher_B"]:
                s[f"gen_correct_{c}"] = sum(G(r, c)["correct"] for r in R)
            s["gen_pass@1_eps_mean"] = float(np.mean([np.mean([G(r, f"self_eps{k}")["correct"] for k in range(K)]) for r in R]))
            s["gen_pass@8"] = sum(any(G(r, f"self_eps{k}")["correct"] for k in range(K)) for r in R)
            s["gen_ended_eps"] = float(np.mean([np.mean([G(r, f"self_eps{k}")["ended"] for k in range(K)]) for r in R]))
            # automatic method proxy: 4-gram overlap closer to E_A or E_B
            lab = lambda g: "A" if g["ov_A"] > g["ov_B"] else ("B" if g["ov_B"] > g["ov_A"] else "=")
            s["gen_auto_method_both_A_and_B_over_eps"] = sum({"A", "B"} <= {lab(G(r, f"self_eps{k}")) for k in range(K)} for r in R)
            s["gen_teacher_auto_switch"] = sum(lab(G(r, "teacher_A")) == "A" and lab(G(r, "teacher_B")) == "B" for r in R)
            s["gen_teacher_identical_AB"] = sum(G(r, "teacher_A")["text"] == G(r, "teacher_B")["text"] for r in R)
        summ[f"{split}/{subset}"] = s
        for k_, v in s.items():
            csv.append(f"{split},{subset},{k_},{json.dumps(v) if isinstance(v, list) else v}")
json.dump(summ, open(f"{O}/evaluations/summary_step{tag}.json", "w"), indent=1)
open(f"{O}/comparisons_step{tag}.csv", "w").write("\n".join(csv) + "\n")
if "gens" in rows[0]:
    with open(f"{O}/heldout_predictions_step{tag}.jsonl", "w") as f:
        for r in rows:
            for g in r["gens"]:
                f.write(json.dumps({"id": r["id"], "role": r["role"], "cond": g["cond"], "noise_id": f"eval:{r['id'][:11]}:{g['cond']}" if g["cond"].startswith("self_eps") else None,
                                    "pred": g["pred"], "correct": g["correct"], "ended": g["ended"], "ov_A": g["ov_A"], "ov_B": g["ov_B"],
                                    "terms_A": g["terms_A"], "terms_B": g["terms_B"], "text": g["text"]}) + "\n")
    with open(f"{O}/evaluations/generations_step{tag}.md", "w") as f:
        for r in sorted(rows, key=lambda r: (r["role"] != "heldout8", r["id"])):
            f.write(f"## {r['id']} ({r['role']})\n\n")
            groups = {}
            for g in r["gens"]:
                groups.setdefault(g["text"], []).append(g)
            for text, gs in groups.items():
                g = gs[0]
                f.write(f"- **{', '.join(x['cond'] for x in gs)}** correct={g['correct']} ov A/B={g['ov_A']:.2f}/{g['ov_B']:.2f} terms A/B={g['terms_A']}/{g['terms_B']}\n\n"
                        f"```\n{text[:700]}\n```\n\n")
print(json.dumps(summ, indent=1))
