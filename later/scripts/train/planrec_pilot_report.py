"""Aggregate the PlanRec pilot: evaluation/metrics.json, evaluation/per_example.csv, evaluation/samples.jsonl,
training curves (probes, losses, gradients) and the A/B batch-order check. Writes no model outputs."""
import argparse
import glob
import json
import os

import pandas as pd


def load_log(path):
    probes, train = {}, []
    for line in open(path):
        d = json.loads(line)
        if any(k.startswith("val/") for k in d):
            probes.setdefault(d["step"], {}).update(d)
        elif "L_total" in d:
            train.append(d)
    return probes, train


def window_means(train, keys, size=50):
    out = []
    for s in range(0, len(train), size):
        w = train[s: s + size]
        out.append({"steps": f"{w[0]['step']}-{w[-1]['step']}",
                    **{k: sum(x[k] for x in w if x.get(k) is not None) / max(1, sum(1 for x in w if x.get(k) is not None))
                       for k in keys}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    ev = os.path.join(a.root, "evaluation")
    runs = {"A": os.path.join(a.root, "run_A_execution_only"), "B": os.path.join(a.root, "run_B_execution_plus_planrec")}
    res = {"models": {}, "runs": {}}
    for f in sorted(glob.glob(os.path.join(ev, "metrics_*.json"))):
        m = json.load(open(f))
        res["models"][m["name"]] = m
    for g, d in runs.items():
        probes, train = load_log(os.path.join(d, "train_log_rank0.jsonl"))
        keys = ["L_execution", "L_planrec", "L_total", "grad_norm_backbone", "grad_norm_latent_projector",
                "grad_norm_total_preclip", "step_time_s", "peak_allocated_GiB", "latent_rms_ratio"]
        z1 = [{k: x[k] for k in x if k.startswith("train/") or k == "step"} for x in train if "train/grad_z1_from_exec" in x]
        res["runs"][g] = {"probes": probes, "loss_windows_50": window_means(train, keys), "z1_grads": z1,
                          "n_steps": len(train), "train_seconds": sum(x["step_time_s"] for x in train),
                          "mean_step_time_s": sum(x["step_time_s"] for x in train) / len(train),
                          "peak_allocated_GiB_max": max(x["peak_allocated_GiB"] for x in train),
                          "examples_seen": 16 * len(train),
                          "exec_tokens_seen_rank0": sum(x["exec_tokens_rank"] for x in train),
                          "planrec_tokens_seen_rank0": sum(x["planrec_tokens_rank"] for x in train),
                          "wandb_run_id": open(os.path.join(d, "wandb_run_id.txt")).read().strip(),
                          "start_provenance": json.load(open(os.path.join(d, "start_provenance.json")))}
    same = True
    for r in (0, 1):
        A = [json.loads(l)["record_ids"] for l in open(os.path.join(runs["A"], f"batch_record_ids_rank{r}.jsonl"))]
        B = [json.loads(l)["record_ids"] for l in open(os.path.join(runs["B"], f"batch_record_ids_rank{r}.jsonl"))]
        same &= A == B
    res["ab_batch_order_identical"] = bool(same)
    res["ab_start_identical"] = all(res["runs"]["A"]["start_provenance"][k] == res["runs"]["B"]["start_provenance"][k]
                                    for k in ("initial_projector_param_hash", "initial_model_param_hash", "seed"))
    json.dump(res, open(os.path.join(ev, "metrics.json"), "w"), indent=1, default=str)
    pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ev, "per_example_*.csv")))]).to_csv(
        os.path.join(ev, "per_example.csv"), index=False)
    with open(os.path.join(ev, "samples.jsonl"), "w") as out:
        for f in sorted(glob.glob(os.path.join(ev, "samples_*.jsonl"))):
            out.writelines(open(f))
    print("REPORT_AGG_DONE", res["ab_batch_order_identical"], res["ab_start_identical"])


if __name__ == "__main__":
    main()
