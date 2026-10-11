"""P0 (read-only): content audit of E2C plans with the frozen rules in plan_span_rules.py.
200 train plans (deterministic sample, seed 42) + the first 128 val plans (pilot probe set)."""
import argparse
import json
import os
import random
import re

import pandas as pd

from later.scripts.data.plan_span_rules import HEADER, build_generic_ngrams, label_chars, save_policy, spans_of
from later.src.train.stage2_common import load_tokenizer
from later.src.train.stage2_plan_dataset import build_views, stage2_tokens, PLAN_OPEN

TAGS = ["generic_template", "method_specific", "subgoal_specific", "unknown"]


def norm(s):
    return re.sub(r"\s+", "", str(s)).lower()


def token_tags(tok, T, row, lab):
    """Tags of the PlanRec target tokens (same tokenization as build_views). A token takes the highest-priority tag
    among its characters (subgoal > method > generic > unknown); whitespace-only parts are ignored."""
    v = build_views(tok, row, 8, T, 4096)
    tgt = [x for x in v["planrec_labels"] if x != -100]
    resp = PLAN_OPEN + row["plan"] + "\n</EXPLORATION><EXECUTION>\n" + row["execution"] + "</EXECUTION><|im_end|>"
    enc = tok(resp, add_special_tokens=False, return_offsets_mapping=True)
    n_open = len(T.plan_open_ids)
    ids, offs = enc["input_ids"][n_open:n_open + len(tgt)], enc["offset_mapping"][n_open:n_open + len(tgt)]
    assert ids == tgt, "PlanRec targets differ from offset tokenization"
    p0, prio = len(PLAN_OPEN), {"subgoal_specific": 3, "method_specific": 2, "generic_template": 1, "unknown": 0}
    out = []
    for a, b in offs:
        cs = [c - p0 for c in range(a, b) if 0 <= c - p0 < len(lab) and not row["plan"][c - p0].isspace()]
        out.append(max((lab[c] for c in cs), key=prio.get) if cs else "generic_template")
    return tgt, offs, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = load_tokenizer(a.stage1)
    T = stage2_tokens(tok)
    tr = pd.read_parquet(f"{a.data_dir}/stage2_train.parquet")
    va = pd.read_parquet(f"{a.data_dir}/stage2_val.parquet")
    gen = build_generic_ngrams(list(tr.plan))
    save_policy(os.path.join(a.out, "span_rules_frozen.json"), gen)
    plan_counts = tr.plan.value_counts()
    sample = sorted(random.Random(42).sample(range(len(tr)), 200))
    sets = [("train", tr.iloc[i].to_dict()) for i in sample] + [("val", va.iloc[i].to_dict()) for i in range(128)]
    rows, ex = [], []
    for split, r in sets:
        plan = r["plan"]
        lab = label_chars(plan, gen)
        tgt, offs, tt = token_tags(tok, T, r, lab)
        nc = sum(1 for c in plan if not c.isspace())
        cc = {t: sum(1 for c, l in zip(plan, lab) if l == t and not c.isspace()) / nc for t in TAGS}
        tc = {t: sum(x == t for x in tt) for t in TAGS}
        ans = str(r["answer"]).strip()
        ans_in_plan = len(norm(ans)) >= 2 and norm(ans) in norm(plan)
        eqs = len(re.findall(r"=\s*-?\d", plan))
        pl = [l.strip() for l in plan.split("\n") if l.strip() and l.strip() != HEADER]
        w = lambda s: re.findall(r"\w+", s.lower())
        pw, ew = w(plan), w(r["execution"])
        eg = {" ".join(ew[i:i + 4]) for i in range(len(ew) - 3)}
        p4 = [" ".join(pw[i:i + 4]) for i in range(len(pw) - 3)]
        d = {"split": split, "record_id": r["record_id"], "plan_tokens": len(tgt), "plan_chars_nonspace": nc,
             **{f"char_frac_{t}": cc[t] for t in TAGS}, **{f"tokens_{t}": tc[t] for t in TAGS},
             "token_frac_generic": tc["generic_template"] / len(tgt),
             "token_frac_specific": (tc["method_specific"] + tc["subgoal_specific"]) / len(tgt),
             "has_method_span": tc["method_specific"] > 0, "has_subgoal_span": tc["subgoal_specific"] > 0,
             "n_method_spans": len(spans_of(lab, "method_specific")), "n_subgoal_spans": len(spans_of(lab, "subgoal_specific")),
             "plan_exact_duplicate_in_train": int(plan_counts.get(plan, 0)) - (1 if split == "train" else 0),
             "answer_string_in_plan_candidate": ans_in_plan, "numeric_equalities_in_plan": eqs,
             "computation_in_plan_candidate": eqs >= 3,
             "plan_lines": len(pl), "plan_lines_restated_in_execution": int(r["n_plan_steps_restated_in_execution"]),
             "plan_4gram_overlap_with_execution": sum(g in eg for g in p4) / max(1, len(p4))}
        rows.append(d)
        ex.append((split, r, lab, tgt, offs, tt))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "p0_plan_audit.csv"), index=False)
    summ = {"n": {"train": int((df.split == "train").sum()), "val": int((df.split == "val").sum())},
            "train_sample_seed": 42, "val_records": "first 128 of stage2_val (pilot / Test-2 probe set)",
            "n_generic_ngrams_from_full_train": len(gen), "by_split": {}}
    for s in ("train", "val"):
        x = df[df.split == s]
        tokens = {t: int(x[f"tokens_{t}"].sum()) for t in TAGS}
        tot = sum(tokens.values())
        summ["by_split"][s] = {
            "plan_tokens_total": tot, "plan_tokens_median": float(x.plan_tokens.median()),
            "token_share": {t: tokens[t] / tot for t in TAGS},
            "char_share_mean": {t: float(x[f"char_frac_{t}"].mean()) for t in TAGS},
            "coverage_records_with_method_span": float(x.has_method_span.mean()),
            "coverage_records_with_subgoal_span": float(x.has_subgoal_span.mean()),
            "coverage_records_with_any_specific": float((x.has_method_span | x.has_subgoal_span).mean()),
            "records_with_specific_share_ge_10pct": float((x.token_frac_specific >= 0.10).mean()),
            "plan_exact_duplicate_rate": float((x.plan_exact_duplicate_in_train > 0).mean()),
            "answer_string_in_plan_candidate_rate": float(x.answer_string_in_plan_candidate.mean()),
            "computation_in_plan_candidate_rate": float(x.computation_in_plan_candidate.mean()),
            "plan_4gram_overlap_with_execution_mean": float(x.plan_4gram_overlap_with_execution.mean()),
            "plan_lines_restated_fraction": float(x.plan_lines_restated_in_execution.sum() / max(1, x.plan_lines.sum()))}
    json.dump(summ, open(os.path.join(a.out, "p0_summary.json"), "w"), indent=1)
    # 24 random examples (seed 7): span text + reason, and token/char alignment check
    pick = random.Random(7).sample(range(len(ex)), 24)
    with open(os.path.join(a.out, "p0_examples.md"), "w") as f:
        f.write("# P0 examples (random, seed 7). Spans by frozen rules; [M]=method, [S]=subgoal, generic in plain, "
                "{unknown} in braces. Token check: decode(tokens tagged specific) vs char spans.\n\n")
        for k in pick:
            split, r, lab, tgt, offs, tt = ex[k]
            plan = r["plan"]
            out, cur = [], None
            for c, l in zip(plan, lab):
                tag = {"method_specific": "M", "subgoal_specific": "S", "unknown": "U", "generic_template": "G"}[l]
                if tag != cur:
                    if cur in ("M", "S"):
                        out.append("]")
                    if cur == "U":
                        out.append("}")
                    if tag in ("M", "S"):
                        out.append(f"[{tag}:")
                    if tag == "U":
                        out.append("{")
                    cur = tag
                out.append(c)
            out.append("]" if cur in ("M", "S") else ("}" if cur == "U" else ""))
            ms = [plan[s:e] for s, e in spans_of(lab, "method_specific")]
            sp_tok = [tok.decode([t]) for t, g in zip(tgt, tt) if g in ("method_specific", "subgoal_specific")]
            f.write(f"## {r['record_id']} ({split})\n\n**Q:** {r['question'][:300]}\n\n```\n{''.join(out)[:1500]}\n```\n\n"
                    f"- method spans (reason: lexicon match): {ms[:8]}\n- specific token share {sum(g in ('method_specific','subgoal_specific') for g in tt)}/{len(tt)}; "
                    f"first specific tokens: {sp_tok[:20]}\n\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
