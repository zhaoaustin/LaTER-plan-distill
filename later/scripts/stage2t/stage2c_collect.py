"""Collect a stage2c_long run into metrics.json / metrics.csv / train_curve.csv (+ step-0 = Stage 2B global-500 full-eval baseline).
usage: python -m later.scripts.stage2t.stage2c_collect RUN_DIR BASELINE_DIR(stage2b_joint_continue500) [RUN_DIR2 ...]"""
import csv
import json
import os
import sys

import numpy as np

STEPS = (0, 50, 100, 150, 200)


def J(p):
    return json.load(open(p)) if os.path.exists(p) else None


def full(d, name):
    acc, ext, summ, dg = (J(f"{d}/evaluations/{k}_{name}.json") for k in ("acc", "extra", "summary", "diag"))
    if acc is None:
        return None
    v, h = ext["val48"], summ["pairs"]["heldout8/all"]
    return {"acc": acc["n_correct"], "acc_ci95": acc["ci95"], "format_ok": acc["format_ok"], "no_boxed": acc["no_boxed"], "n": acc["n"],
            "exec_ce_val48": v["gold_ce_self"], "dCE_zero_val48": v["dCE_zero"], "dCE_swap_val48": v["dCE_crossq"],
            "jsd_with_blind_val48_T": v["jsd_aware_blind_T"]["mean"], "jsd_with_swap_val48_T": v["jsd_self_crossq_T"]["mean"],
            "fr_paired": dg["fr"]["paired"], "fr_gap_perm": dg["fr"]["gap_perm"], "zs_crossq_cos": dg["collapse"]["zs_crossq_cos"],
            "var_ratio": dg["collapse"]["zs_var_over_teacher_var"], "var_ratio_per_slot": dg["collapse"]["zs_var_over_teacher_var_per_slot"],
            "oracle_heldout8_matched_minus_swapped": ext["heldout8"]["full_matched_minus_swapped"], "oracle_heldout8_both_correct": ext["heldout8"]["full_both_correct"],
            "oracle_heldout8_dspan_both": [ext["heldout8"]["dspan_both_correct"], ext["heldout8"]["dspan_n"]],
            "gen_teacher_auto_switch_heldout8": h.get("gen_teacher_auto_switch"), "gen_TA_identical_TB": h.get("gen_TA_identical_TB"),
            "gen_auto_method": {c: h.get(f"gen_auto_method_{c}") for c in ("self", "crossq", "mean", "zero", "teacher_A", "teacher_B")},
            "gen_correct": {c: h.get(f"gen_correct_{c}") for c in ("self", "crossq", "mean", "zero", "teacher_A", "teacher_B")}}


def collect(d, base):
    tr = [json.loads(l) for l in open(f"{d}/train.jsonl")] if os.path.exists(f"{d}/train.jsonl") else []
    meta = J(f"{d}/run_meta.json")
    with open(f"{d}/train_curve.csv", "w", newline="") as f:
        keys = ["step", "distill_jsd", "anchor_jsd_pos", "anchor_weighted", "total_loss", "grad_norm_projector", "grad_norm_lora", "grad_total_preclip", "clipped", "lr", "finite", "exec_tokens", "step_s"]
        w = csv.writer(f); w.writerow(keys)
        for r in tr:
            w.writerow([r[k] for k in keys])
    rows = []
    for s in STEPS:
        L = J(f"{d}/evaluations/light_step{s}.json")
        win = [r for r in tr if s - 9 <= r["step"] <= s] if s else []
        m = lambda k: float(np.mean([r[k] for r in win])) if win else None
        row = {"step": s, "train_distill_jsd_win10": m("distill_jsd"), "train_anchor_jsd_win10": m("anchor_jsd_pos"), "train_total_win10": m("total_loss"),
               "train_gn_win10": m("grad_total_preclip")}
        if L:
            row.update({k: L[k] for k in ("jsd_with_blind", "jsd_with_swap", "nll_self", "dnll_zero", "dnll_zero_frac_pos", "dnll_swap", "dnll_swap_frac_pos", "dnll_drop",
                                          "gen_correct", "gen_format_ok", "gen_empty", "gen_degenerate", "severe_collapse", "teacher_unchanged")})
            for k, v in (L.get("diag48") or {}).items():
                row["diag48_" + k] = v
        rows.append(row)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "step", k))
    with open(f"{d}/metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); [w.writerow(r) for r in rows]
    out = {"run": d, "lam_anchor": meta and meta["lam_anchor"], "n_train_steps_logged": len(tr),
           "train_summary": {"max_grad_total_preclip": max((r["grad_total_preclip"] for r in tr), default=None), "n_clipped": sum(r["clipped"] for r in tr),
                             "all_finite": all(r["finite"] for r in tr), "all_aligned": all(r["labels_aligned"] for r in tr), "teacher_grad_none_all": all(r["teacher_grad_none"] for r in tr),
                             "train_minutes": (tr[-1]["elapsed_s"] / 60) if tr else None, "mean_step_s": float(np.mean([r["step_s"] for r in tr])) if tr else None,
                             "peak_mem_gb": max((r["peak_mem_gb"] for r in tr), default=None)},
           "steps": rows, "full_step0_baseline_stage2b_500": full(base, "joint_step500"), "full_step200": full(d, "stage2c_step200")}
    json.dump(out, open(f"{d}/metrics.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    base = sys.argv[2]
    for d in [sys.argv[1]] + sys.argv[3:]:
        o = collect(d, base); print(d, json.dumps({k: o[k] for k in ("lam_anchor", "n_train_steps_logged", "train_summary")}))
