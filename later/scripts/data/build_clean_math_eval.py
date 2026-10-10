"""Fixed, small, contamination-checked math eval set for Stage-2 monitoring.

MATH-500 problems are kept only if they share no normalized exact match and no 13-word n-gram with ANY row of the
full E2C SFT source parquet (a superset of what the Stage-1 checkpoint was trained on, so these problems were not
seen in Stage 1 or Stage 2). Fixed seed shuffle, first N kept.
"""
import argparse
import hashlib
import json
import random
import re

import pandas as pd


def norm_q(q):
    return re.sub(r"\s+", " ", str(q).strip().lower())


def ngrams(s, n=13):
    w = re.findall(r"\w+", str(s).lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--math500", required=True)
    ap.add_argument("--sft_source", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sft = pd.read_parquet(args.sft_source)
    sft_q = set(sft.question.map(norm_q))
    sft_ng = set()
    for q in sft.question:
        sft_ng |= ngrams(q)
    m5 = pd.read_parquet(args.math500)
    idx = list(range(len(m5)))
    random.Random(args.seed).shuffle(idx)
    keep, skipped = [], {"exact": 0, "13gram": 0}
    for i in idx:
        q = m5.question.iloc[i]
        if norm_q(q) in sft_q:
            skipped["exact"] += 1
        elif ngrams(q) & sft_ng:
            skipped["13gram"] += 1
        else:
            keep.append(i)
        if len(keep) == args.n:
            break
    out = pd.DataFrame({"problem_id": [f"math500-{i}" for i in keep], "question": m5.question.iloc[keep].tolist(),
                        "answer": m5.answer.iloc[keep].tolist()})
    out.to_parquet(args.out, index=False)
    meta = {"source": args.math500, "sft_source": args.sft_source,
            "sft_source_sha256": hashlib.sha256(open(args.sft_source, "rb").read()).hexdigest(),
            "seed": args.seed, "n": len(out), "scanned": idx.index(keep[-1]) + 1, "skipped_contaminated": skipped,
            "method": "drop if normalized exact match or any shared 13-word n-gram with the E2C SFT source",
            "residual_risk": "paraphrased duplicates are not detected by 13-gram matching",
            "problem_ids": out.problem_id.tolist()}
    json.dump(meta, open(args.out.replace(".parquet", ".json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("n", "scanned", "skipped_contaminated")}))


if __name__ == "__main__":
    main()
