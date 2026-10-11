"""dataset_manifest.json for the PlanRec pilot: hashes, counts, lossless Stage-1 reconstruction over every record,
PlanRec / execution target statistics under the exact build_views tokenization, contamination notes."""
import argparse
import json
import statistics as st

import pandas as pd

from later.src.train.stage2_common import checkpoint_hash, dump_json, file_sha256, load_tokenizer, tokenizer_hashes
from later.src.train.stage2_plan_dataset import (BOUNDARY_RULE_VERSION, EXEC_CLOSE, PLAN_CONTENT_RULE, PLAN_OPEN,
                                                 PLANREC_RULE_VERSION, SEPARATOR, Stage2RecordError, build_views,
                                                 stage2_tokens)


def q(xs):
    xs = sorted(xs)
    return {"min": xs[0], "p50": xs[len(xs) // 2], "p95": xs[int(0.95 * len(xs))], "max": xs[-1], "mean": st.mean(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = load_tokenizer(a.stage1)
    T = stage2_tokens(tok)
    audit = json.load(open(f"{a.data_dir}/stage1_to_stage2_data_audit.json"))
    man = {"stage1_checkpoint": {"path": a.stage1, "hf_repo": "zhaohq/Qwen3-4B-E2C-SFT-only",
                                 "hf_revision": "a6b0e2fe008f14b00bf7f3a29a214e96212f7251",
                                 "safetensors_sha256": {f: file_sha256(f"{a.stage1}/{f}") for f in
                                                        ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")},
                                 "matches_hf_revision_lfs_sha256": True, "checkpoint_hash": checkpoint_hash(a.stage1)},
           **tokenizer_hashes(tok, a.stage1), "boundary_rule_version": BOUNDARY_RULE_VERSION,
           "planrec_rule_version": PLANREC_RULE_VERSION, "plan_content_rule": PLAN_CONTENT_RULE,
           "stage1_source": {"paths": audit["stage1_dataset_paths"], "sha256": audit["sha256"], "n_source": audit["n_source"],
                             "n_parsed": audit["n_parsed"], "n_accepted": audit["n_accepted"],
                             "rejected_by_reason": audit["n_rejected_by_reason"]},
           "data_source_policy": "stage1_sft_only", "later69k_samples_used": 0, "new_teacher_data": 0, "splits": {}}
    for split in ("train", "val"):
        path = f"{a.data_dir}/stage2_{split}.parquet"
        f = pd.read_parquet(path)
        lossless, errs, n_plan, n_cont, n_math, n_exec, n_stud = 0, {}, [], [], [], [], []
        for r in f.to_dict("records"):
            if PLAN_OPEN + r["plan"] + SEPARATOR + r["execution"] + EXEC_CLOSE == r["stage1_solution"]:
                lossless += 1
            try:
                v = build_views(tok, r, 8, T, 4096)
            except Stage2RecordError as e:
                errs[str(e)] = errs.get(str(e), 0) + 1
                continue
            n_plan.append(sum(x != -100 for x in v["planrec_labels"]))
            n_cont.append(sum(v["planrec_content"]))
            n_math.append(sum(v["planrec_math"]))
            n_exec.append(sum(x != -100 for x in v["student_labels"]))
            n_stud.append(len(v["student_ids"]))
        man["splits"][split] = {"path": path, "sha256": file_sha256(path), "n_records": len(f),
                                "record_ids_unique": bool(f.record_id.is_unique),
                                "stage1_solution_lossless": lossless, "build_views_errors": errs,
                                "planrec_target_tokens": q(n_plan), "planrec_content_tokens": q(n_cont),
                                "planrec_math_tokens": q(n_math), "execution_target_tokens": q(n_exec),
                                "student_sequence_tokens": q(n_stud),
                                "content_fraction_of_plan_tokens": sum(n_cont) / sum(n_plan)}
        print(split, json.dumps({k: v for k, v in man["splits"][split].items() if k in ("n_records", "stage1_solution_lossless", "build_views_errors")}), flush=True)
    man["contamination"] = {
        "stage2_data": "train/val records overlapping AIME24/25, MATH-500, OlympiadBench, AMC23 (exact or 13-gram) were "
                       "removed by the Stage-1->Stage-2 conversion (eval_overlap: train 301, val 19); val rows whose "
                       "question also appears in train removed (7).",
        "stage1_checkpoint": "the Stage-1 SFT checkpoint itself was trained on 320 benchmark-overlapping records.",
        "gen_eval_set": "data/clean_math_eval.parquet: 50 MATH-500 problems, 13-gram clean against the full E2C SFT "
                        "source; paraphrases are not detected.",
        "probe_set": "first 128 (training probes) / 200 (final eval) records of stage2_val, fixed."}
    man["closing_token_note"] = ("stage1_solution ends with '</EXECUTION>'; the Stage-1 training target appends "
                                 "'<|im_end|>' (build_views checks response == stage1_solution + '<|im_end|>'). "
                                 "The separator '\\n</EXPLORATION><EXECUTION>\\n' contains the merged token '><' (id "
                                 "1784); its leading '\\n' is part of the Stage-1 plan segment, so PlanRec targets "
                                 "decode to plan + '\\n'.")
    dump_json(man, a.out)
    print("MANIFEST_DONE")


if __name__ == "__main__":
    main()
