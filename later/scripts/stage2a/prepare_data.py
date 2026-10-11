"""Stage 2A data: plan-only samples from the existing E2C Stage-2 parquet (train/val), question-level disjoint,
excluding the 28 bridge dual-plan questions and exact (normalized) matches of downstream benchmark problems.
Plans: the <EXPLORATION> plan field; normalization removes ONLY the global header line
'I need to make a step-by-step plan.' (identical for 100% of records). Plans longer than max_plan_tokens (decoder
tokenizer, + EOS) are filtered (counted), never truncated."""
import argparse, glob, hashlib, json, os, re
import numpy as np, pandas as pd
from transformers import AutoTokenizer

HEADER = "I need to make a step-by-step plan.\n"


def nq(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def norm_plan(p):
    return p[len(HEADER):] if p.startswith(HEADER) else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--bench_dir", required=True)
    ap.add_argument("--bridge_dir", required=True); ap.add_argument("--dec_tok", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n_train", type=int, default=3000); ap.add_argument("--n_val", type=int, default=300)
    ap.add_argument("--max_plan_tokens", type=int, default=288); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.dec_tok)
    tr = pd.read_parquet(f"{a.data_dir}/stage2_train.parquet"); va = pd.read_parquet(f"{a.data_dir}/stage2_val.parquet")
    man = {"source": {"train": f"{a.data_dir}/stage2_train.parquet", "val": f"{a.data_dir}/stage2_val.parquet",
                      "n_train_records": len(tr), "n_val_records": len(va)}}
    # benchmarks
    bench = {}
    for f in sorted(glob.glob(f"{a.bench_dir}/*.parquet")):
        b = pd.read_parquet(f)
        col = next((c for c in ("problem", "question", "prompt", "Problem", "Question") if c in b.columns), None)
        if col is None:
            continue
        vals = b[col].apply(lambda x: x if isinstance(x, str) else (x[0]["content"] if hasattr(x, "__len__") and len(x) and isinstance(x[0], dict) else str(x)))
        bench[os.path.basename(f)] = {nq(v) for v in vals}
    bench_all = set().union(*bench.values())
    bridge = [json.loads(l) for f in ("train_pairs.jsonl", "heldout_pairs.jsonl") for l in open(f"{a.bridge_dir}/{f}")]
    bridge_q = {nq(p["question"]) for p in bridge}
    held = [json.loads(l) for l in open(f"{a.bridge_dir}/heldout_pairs.jsonl")]
    stats = {}
    def prep(df, split):
        d = df.copy()
        d["qn"] = d.question.map(nq)
        d["question_hash"] = d.qn.map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:16])
        d["plan_norm"] = d.plan.map(norm_plan)
        d["n_tok"] = d.plan_norm.map(lambda s: len(tok.encode(s)) + 1)
        n0 = len(d)
        d = d[d.plan_norm.str.strip().str.len() > 0]
        dup = d.qn.duplicated(keep="first"); d = d[~dup]
        inb = d.qn.isin(bench_all); d_b = int(inb.sum()); d = d[~inb]
        inbr = d.qn.isin(bridge_q); d_br = int(inbr.sum()); d = d[~inbr]
        too_long = d.n_tok > a.max_plan_tokens
        stats[split] = {"records": n0, "dup_questions_removed": int(dup.sum()), "benchmark_exact_matches_removed": d_b,
                        "bridge_questions_removed": d_br, "over_max_tokens_filtered": int(too_long.sum()),
                        "over_max_tokens_frac": float(too_long.mean()), "eligible": int((~too_long).sum()),
                        "plan_tokens_p50_p90_p95_max_before_filter": [float(x) for x in np.percentile(d.n_tok, [50, 90, 95])] + [int(d.n_tok.max())]}
        return d[~too_long]
    trd, vad = prep(tr, "train"), prep(va, "val")
    vad = vad[~vad.qn.isin(set(trd.qn))]
    trs = trd.sample(a.n_train, random_state=a.seed).sort_index()
    vas = vad.sample(a.n_val, random_state=a.seed).sort_index()
    assert not set(trs.qn) & set(vas.qn) and not set(trs.qn) & bridge_q and not set(vas.qn) & bridge_q
    cols = ["record_id", "question_hash", "plan", "plan_norm", "n_tok"]
    for name, d in (("train", trs), ("val", vas)):
        with open(f"{a.out}/{name}_plans.jsonl", "w") as f:
            for _, r in d.iterrows():
                f.write(json.dumps({**{c: (int(r[c]) if c == "n_tok" else r[c]) for c in cols}, "split": name}) + "\n")
    # held-out dual plans (P_A/P_B only used) + extended set (the 20 bridge-train pairs, also unseen by 2A)
    with open(f"{a.out}/paired_plans.jsonl", "w") as f:
        for p in bridge:
            role = "heldout8" if p["id"] in {h["id"] for h in held} else "extended20"
            f.write(json.dumps({"id": p["id"], "role": role, "question_hash": hashlib.sha1(nq(p["question"]).encode()).hexdigest()[:16],
                                "plan_A_norm": norm_plan(p["plan_A"]), "plan_B_norm": norm_plan(p["plan_B"]),
                                "method_label_A": p.get("method_label_A"), "method_label_B": p.get("method_label_B"),
                                "method_terms_only_A": p["method_terms_only_A"], "method_terms_only_B": p["method_terms_only_B"],
                                "n_tok_A": len(tok.encode(norm_plan(p["plan_A"]))) + 1, "n_tok_B": len(tok.encode(norm_plan(p["plan_B"]))) + 1}) + "\n")
    held_in_train = sum(nq(h["question"]) in set(tr.question.map(nq)) for h in held)
    man.update({"stats": stats, "n_train": len(trs), "n_val": len(vas), "seed": a.seed, "max_plan_tokens": a.max_plan_tokens,
                "max_plan_tokens_note": "task default 192 would filter ~28% of plans (p90=245); raised to 288 (~p95) per task rule",
                "benchmarks_checked": {k: len(v) for k, v in bench.items()}, "benchmark_check": "exact match after lowercase+whitespace normalization; near-duplicates NOT detected",
                "bridge_heldout8_questions_in_stage2_train": int(held_in_train),
                "bridge_pairs_source": "stage2_val rows 256-655 (Stage-1 samples, verified)", "train_val_question_overlap": 0,
                "plan_normalization": "strip leading header line only", "files_sha256": {f: hashlib.sha256(open(f"{a.out}/{f}", "rb").read()).hexdigest()
                                                                                       for f in ("train_plans.jsonl", "val_plans.jsonl", "paired_plans.jsonl")}})
    json.dump(man, open(f"{a.out}/data_manifest.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != "files_sha256"}, indent=1))


if __name__ == "__main__":
    main()
