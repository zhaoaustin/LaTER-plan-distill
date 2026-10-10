"""Build Stage-2 (plan distillation) records from OUR Stage-1 explicit-planning SFT data.

Source = the exact parquet files the Stage-1 checkpoint was trained/validated on. Each Stage-1 target is
    <EXPLORATION>\n{P}\n</EXPLORATION><EXECUTION>\n{E}</EXECUTION>      (+ "<|im_end|>" appended at training)
and the answer A is the content of the last balanced \\boxed{...} inside E (E2C keeps the answer inside the
execution). Parsing is exact-structure only; a record is accepted only if the parts reassemble to the original
target byte-for-byte. Nothing is regenerated or rewritten. No LaTER-69K data is read.

Outputs: stage2_{train,val}.parquet with stable record_id + provenance, and stage1_to_stage2_data_audit.json.
"""
import argparse
import hashlib
import json
import os
import random
import re

import pandas as pd

EB, EE, XB, XE = "<EXPLORATION>\n", "\n</EXPLORATION><EXECUTION>\n", None, "</EXECUTION>"
EVAL_SETS = ["aime24", "aime25", "amc23", "gsm8k", "math500", "minerva", "olympiad_bench"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def norm_q(q):
    return re.sub(r"\s+", " ", str(q).strip().lower())


def ngrams(s, n=13):
    w = re.findall(r"\w+", str(s).lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None, -1
    j = s.find("{", i)
    if j < 0:
        return None, -1
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k], i
    return None, -1


def restates_answer(plan, ans, question):
    """Plan states the final answer: answer (>=4 non-space chars) occurs in the plan as a standalone expression
    (whitespace-flexible, not glued to letters/digits) and does not already occur in the question."""
    a = re.sub(r"\s", "", ans)
    if len(a) < 4:
        return False
    pat = re.compile(r"(?<![A-Za-z0-9])" + r"\s*".join(map(re.escape, a)) + r"(?![A-Za-z0-9])")
    return bool(pat.search(plan)) and not pat.search(question)


def parse(solution):
    """Return (P, E, A, a_char_start_in_E) or raise ValueError(reason)."""
    if not solution.startswith(EB):
        raise ValueError("missing_exploration_open")
    if solution.count(EE) != 1:
        raise ValueError("plan_exec_separator_not_unique")
    if not solution.endswith(XE) or solution.count(XE) != 1:
        raise ValueError("execution_close_not_unique_or_not_final")
    plan, rest = solution[len(EB):].split(EE, 1)
    execution = rest[: -len(XE)]
    if EB + plan + EE + execution + XE != solution:
        raise ValueError("reassembly_mismatch")
    if not plan.strip():
        raise ValueError("empty_plan")
    if not execution.strip():
        raise ValueError("empty_execution")
    for tag in ("<EXPLORATION>", "</EXPLORATION>", "<EXECUTION>", "</EXECUTION>"):
        if tag in plan or tag in execution:
            raise ValueError("nested_tag")
    ans, a_start = last_boxed(execution)
    if ans is None or not ans.strip():
        raise ValueError("no_parsable_answer")
    return plan, execution, ans, a_start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_train", required=True)
    ap.add_argument("--stage1_val", required=True)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # benchmark questions for contamination check
    eval_q, eval_ng = {}, {}
    for name in EVAL_SETS:
        p = os.path.join(args.eval_dir, f"{name}.parquet")
        if os.path.exists(p):
            qs = pd.read_parquet(p).question.tolist()
            eval_q[name] = {norm_q(q) for q in qs}
            ng = set()
            for q in qs:
                ng |= ngrams(q)
            eval_ng[name] = ng

    audit = {"stage1_dataset_paths": {}, "sha256": {}, "n_source": {}, "n_parsed": {}, "n_accepted": {},
             "n_rejected_by_reason": {}, "checkpoint_path": args.checkpoint,
             "data_source_policy": "stage1_sft_only", "later69k_samples_used": 0}
    frames = {}
    for split, path in [("train", args.stage1_train), ("val", args.stage1_val)]:
        df = pd.read_parquet(path)
        audit["stage1_dataset_paths"][split] = os.path.abspath(path)
        audit["sha256"][split] = sha256(path)
        audit["n_source"][split] = len(df)
        recs, rej = [], {}
        for idx, (q, sol) in enumerate(zip(df.question.tolist(), df.solution.tolist())):
            rid = f"s1{split}-{idx:05d}-{hashlib.sha1((q + sol).encode()).hexdigest()[:8]}"
            try:
                if not str(q).strip():
                    raise ValueError("empty_question")
                plan, execution, ans, a_start = parse(sol)
            except ValueError as e:
                rej[str(e)] = rej.get(str(e), 0) + 1
                continue
            reasons = []
            if "\\boxed" in plan:
                reasons.append("plan_contains_boxed_answer")
            if restates_answer(plan, ans, q):
                reasons.append("plan_restates_final_answer")
            qn = norm_q(q)
            hits = [n for n in eval_q if qn in eval_q[n]] or [n for n in eval_ng if ngrams(q) & eval_ng[n]]
            if hits:
                reasons.append("eval_overlap:" + "+".join(sorted(hits)))
            if reasons:
                for r in reasons:
                    key = r.split(":")[0] if r.startswith("eval_overlap") else r
                    rej[key] = rej.get(key, 0) + 1
                    if r.startswith("eval_overlap"):
                        audit.setdefault("eval_overlap_detail", {}).setdefault(split, []).append(
                            {"record_id": rid, "benchmarks": hits})
                continue
            plan_lines = [l.strip() for l in plan.split("\n") if re.match(r"^\s*\d+\.", l)]
            copied = sum(1 for l in plan_lines if l in execution)
            recs.append({"record_id": rid, "split": split, "source_file": os.path.abspath(path),
                         "source_row": idx, "question": q, "plan": plan, "execution": execution, "answer": ans,
                         "answer_char_start_in_execution": a_start, "stage1_solution": sol,
                         "n_plan_steps": len(plan_lines), "n_plan_steps_restated_in_execution": copied})
        audit["n_parsed"][split] = len(df) - sum(v for k, v in rej.items() if k in (
            "missing_exploration_open", "plan_exec_separator_not_unique", "execution_close_not_unique_or_not_final",
            "reassembly_mismatch", "empty_plan", "empty_execution", "nested_tag", "no_parsable_answer",
            "empty_question"))
        audit["n_rejected_by_reason"][split] = rej
        frames[split] = pd.DataFrame(recs)

    # group isolation: drop val records whose question also occurs in train
    tr_q = set(frames["train"].question.map(norm_q))
    leak = frames["val"].question.map(norm_q).isin(tr_q)
    audit["n_rejected_by_reason"]["val"]["question_also_in_train"] = int(leak.sum())
    frames["val"] = frames["val"][~leak].reset_index(drop=True)
    for split in ("train", "val"):
        audit["n_accepted"][split] = len(frames[split])
        frames[split].to_parquet(os.path.join(args.out_dir, f"stage2_{split}.parquet"), index=False)
    tq = frames["train"].question.map(norm_q)
    audit.update({
        "n_train": len(frames["train"]), "n_val": len(frames["val"]),
        "n_unique_questions": {"train": int(tq.nunique()), "val": int(frames["val"].question.map(norm_q).nunique())},
        "n_duplicate_questions": {"train_rows_with_repeated_question": int(len(tq) - tq.nunique()),
                                  "train_val_shared_after_filter": 0},
        "n_overlap_eval": {s: len(audit.get("eval_overlap_detail", {}).get(s, [])) for s in ("train", "val")},
        "eval_overlap_method": "normalized exact question match, else any shared 13-word n-gram, against "
                               + ", ".join(eval_q) + " (E2C official eval parquets)",
        "stage1_chat_template": "Qwen3 tokenizer.apply_chat_template([{'role':'user','content':Q}], "
                                "add_generation_prompt=True) with no enable_thinking kwarg == "
                                "'<|im_start|>user\\n{Q}<|im_end|>\\n<|im_start|>assistant\\n'",
        "stage1_target_template": "<EXPLORATION>\\n{P}\\n</EXPLORATION><EXECUTION>\\n{E}</EXECUTION><|im_end|>",
        "recovered_fields": {"question": "parquet column 'question'", "plan": "text between '<EXPLORATION>\\n' "
                             "and '\\n</EXPLORATION><EXECUTION>\\n'", "execution": "text between that separator "
                             "and the final '</EXECUTION>'", "answer": "content of the last balanced \\boxed{} in "
                             "execution (answer stays inside E, as in Stage 1)"},
        "answer_correctness": "Stage-1 data has no gold-answer column; correctness of A cannot be verified, A is "
                              "exactly the Stage-1 target's answer",
        "plan_step_restatement": {s: {"mean_plan_steps": float(frames[s].n_plan_steps.mean()),
                                      "frac_steps_restated": float(frames[s].n_plan_steps_restated_in_execution.sum()
                                                                   / max(1, frames[s].n_plan_steps.sum()))}
                                  for s in ("train", "val")},
    })
    json.dump(audit, open(os.path.join(args.out_dir, "stage1_to_stage2_data_audit.json"), "w"), indent=1,
              ensure_ascii=False)
    print(json.dumps({k: audit[k] for k in ["n_source", "n_parsed", "n_accepted", "n_rejected_by_reason",
                                            "n_overlap_eval", "plan_step_restatement"]}, indent=1))


if __name__ == "__main__":
    main()
