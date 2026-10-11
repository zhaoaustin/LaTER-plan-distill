"""Aggregate the A/B/C/D bridge-input runs (CPU only): metrics json, held-out per-pair jsonl, generations md, manifest."""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

O, MINI, REPO = sys.argv[1], sys.argv[2], sys.argv[3]
G = {"A_raw": "raw", "B_std": "std", "C_res": "res", "D_res_std": "res_std"}
held = [json.loads(l) for l in open(os.path.join(O, "heldout_pairs.jsonl"))]
train = [json.loads(l) for l in open(os.path.join(O, "train_pairs.jsonl"))]
orc_m = {o["id"]: o for o in json.load(open(os.path.join(MINI, "final_metrics.json")))["oracle_text_plan"]}
orc_g = {}
for l in open(os.path.join(MINI, "heldout_predictions.jsonl")):
    r = json.loads(l)
    if r["condition"].startswith("oracle"):
        orc_g[(r["id"], r["condition"])] = r
metrics = {"groups": {}, "oracle_text_plan_full_margin_mean": json.load(open(os.path.join(MINI, "final_metrics.json")))["oracle_mean_margin"],
           "old_A_mini_pilot_step50_full_margin": json.load(open(os.path.join(MINI, "final_metrics.json")))["heldout"]["mean_margin"]}
per_pair = []
for g, m in G.items():
    d = os.path.join(O, f"group_{m}")
    log = [json.loads(l) for l in open(os.path.join(d, "train_log.jsonl"))]
    ev = {x["step"]: x for x in log if x.get("eval")}
    tr = [x for x in log if not x.get("eval")]
    gm = {"curve_heldout_margin_span": {s: e["heldout"]["mean_margin_span"] for s, e in ev.items()},
          "curve_train_margin_span": {s: e["train"]["mean_margin_span"] for s, e in ev.items()},
          "curve_heldout_zcos": {s: e["heldout"]["z_cos_AB"] for s, e in ev.items()},
          "step_time_s": tr[-1]["elapsed_s"] / len(tr), "peak_mem_gb": tr[-1]["peak_mem_gb"],
          "audit": {k: v for k, v in json.load(open(os.path.join(d, "input_representation_audit.json"))).items() if k != "boundary_literal"},
          "boundary_literal": {k: v for k, v in json.load(open(os.path.join(d, "input_representation_audit.json")))["boundary_literal"].items() if k != "per_pair"},
          "smoke": json.load(open(os.path.join(d, "unit_tests_smoke.json")))["ALL_HARD_GATES_PASS"]}
    gm["step0"] = ev[0]["heldout"]
    for s in (50, 100):
        f = json.load(open(os.path.join(d, f"final_step{s}.json")))
        gens = [json.loads(l) for l in open(os.path.join(d, f"generations_step{s}.jsonl"))]
        gg = {(x["id"], x["condition"]): x for x in gens}
        sw = []
        for p in held:
            a, b = gg[(p["id"], "Z_A")], gg[(p["id"], "Z_B")]
            # automatic switch proxy: Z_A generation closer to E_A and Z_B generation closer to E_B (4-gram overlap)
            sw.append({"identical": a["generation"] == b["generation"],
                       "auto_switch": a["overlap_E_A"] > a["overlap_E_B"] and b["overlap_E_B"] > b["overlap_E_A"],
                       "correct_A": a["correct"], "correct_B": b["correct"]})
        gm[f"step{s}"] = {"heldout": f["heldout"], "heldout_margin_span_ci95": f["heldout_margin_span_ci95"], "train": f["train"],
                          "gen_identical_pairs": sum(x["identical"] for x in sw), "gen_auto_switch_pairs": sum(x["auto_switch"] for x in sw),
                          "gen_correct": sum(x["correct_A"] + x["correct_B"] for x in sw)}
        for r in f["heldout_rows"]:
            per_pair.append({"group": g, "step": s, **r})
    metrics["groups"][g] = gm
json.dump(metrics, open(os.path.join(O, "bridge_input_abcd_metrics.json"), "w"), indent=1)
with open(os.path.join(O, "heldout_per_pair.jsonl"), "w") as f:
    for r in per_pair:
        f.write(json.dumps(r) + "\n")
# generations md (step 50 = main comparison)
with open(os.path.join(O, "heldout_generations.md"), "w") as f:
    f.write("# Held-out greedy generations, step 50 (max 1024 new tokens). Oracle = Stage-1 with the TEXT plan "
            "(reused from the mini pilot, max 1536). Auto metrics: answer correctness (math_verify), 4-gram overlap with E_A / E_B, "
            "lexicon method-term hits (A/B). Method judgment in ABCD_REPORT.md.\n\n")
    for p in held:
        f.write(f"## {p['id']}\n\n**Q:** {p['question'][:500]}\n\n**Method A:** {p['method_label_A']}  \n**Method B:** {p['method_label_B']}\n\n"
                f"<details><summary>P_A / P_B</summary>\n\n```\n{p['plan_A']}\n```\n```\n{p['plan_B']}\n```\n</details>\n\n")
        for cond in ("oracle_text_PA", "oracle_text_PB"):
            r = orc_g[(p["id"], cond)]
            f.write(f"- **{cond}**: correct={r['correct']}, overlap E_A/E_B={r['overlap_with_E_A']:.2f}/{r['overlap_with_E_B']:.2f}, "
                    f"terms A/B={r['method_terms_A_hits']}/{r['method_terms_B_hits']}\n")
        for g, m in G.items():
            gens = {x["condition"]: x for x in map(json.loads, open(os.path.join(O, f"group_{m}", "generations_step50.jsonl"))) if x["id"] == p["id"]}
            for cond in ("Z_A", "Z_B"):
                r = gens[cond]
                f.write(f"\n<details><summary>{g} {cond}: correct={r['correct']}, overlap E_A/E_B={r['overlap_E_A']:.2f}/{r['overlap_E_B']:.2f}, "
                        f"terms A/B={r['method_terms_A_hits']}/{r['method_terms_B_hits']}"
                        f"{' (IDENTICAL to Z_A)' if cond == 'Z_B' and r['generation'] == gens['Z_A']['generation'] else ''}</summary>\n\n"
                        f"```\n{r['generation'][:3000]}\n```\n</details>\n")
        f.write("\n")
# train stats bundle
np.savez(os.path.join(O, "train_stats.npz"), **{f"{m}_{k}": v for m in ("std", "res_std")
                                                 for k, v in np.load(os.path.join(O, f"group_{m}", "train_stats.npz")).items()})
git = lambda *x: subprocess.run(["git", "-C", REPO, *x], capture_output=True, text=True).stdout.strip()
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
manifest = {"repo_branch": git("branch", "--show-current"), "repo_head": git("rev-parse", "HEAD"),
            "base_commit_bridge_code": "d407338 (exp/bridge-mini-pilot-1011)",
            "model": "/scratch/hz33227/e2c/Explore-Execute-Chain/outputs/e2c_qwen3_4b_sft/best_hf = Stage-1 step 800 = HF zhaohq/Qwen3-4B-E2C-SFT-only @ a6b0e2fe008f14b00bf7f3a29a214e96212f7251",
            "model_files_sha256": {k: v["sha256"] for k, v in json.load(open("/scratch/hz33227/e2c/Explore-Execute-Chain/outputs/e2c_qwen3_4b_sft/artifact_manifest.json"))["files"].items() if "safetensors" in k or "tokenizer" in k},
            "data": {"train_pairs_sha256": sha(os.path.join(O, "train_pairs.jsonl")), "heldout_pairs_sha256": sha(os.path.join(O, "heldout_pairs.jsonl")),
                     "train_ids": [p["id"] for p in train], "heldout_ids": [p["id"] for p in held], "removed_pairs": 0},
            "shared_config": {"seed_init": 0, "data_order_seed": 1234, "pairs_per_step": 4, "lr": 1e-4, "weight_decay": 0.01, "grad_clip": 1.0,
                              "lambda_rank": 0.1, "rank_span_tokens": 64, "enc_layer": 27, "slots": 8, "steps": 100, "eval_every": 10,
                              "std_clip": 5.0, "sigma_floor_percentile": 5.0, "gen_max_new_tokens": 1024},
            "gpu": "2x NVIDIA B200 (Slurm job 48907334, node e4-1); groups run 2 per GPU sequentially",
            "torch": "2.8.0+cu128", "transformers": "5.12.1"}
json.dump(manifest, open(os.path.join(O, "run_manifest.json"), "w"), indent=1)
for g, gm in metrics["groups"].items():
    s5, s10 = gm["step50"], gm["step100"]
    print(g, "| s50 held span %.5f %s frac %.2f both %d full %.5f | zcos %.4f | train span %.5f | gen ident %d autosw %d corr %d | s100 held span %.5f train %.5f autosw %d" % (
        s5["heldout"]["mean_margin_span"], [round(x, 5) for x in s5["heldout_margin_span_ci95"]], s5["heldout"]["frac_pos_span"],
        s5["heldout"]["pairs_both_pos_span"], s5["heldout"]["mean_margin_full"], s5["heldout"]["z_cos_AB"], s5["train"]["mean_margin_span"],
        s5["gen_identical_pairs"], s5["gen_auto_switch_pairs"], s5["gen_correct"], s10["heldout"]["mean_margin_span"],
        s10["train"]["mean_margin_span"], s10["gen_auto_switch_pairs"]))
