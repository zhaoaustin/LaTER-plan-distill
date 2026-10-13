"""Data for the dual-initialization bridge pilot (one shared data set for both runs).

candidates: same auto filter as the mini bridge pilot (build_pairs.py / candidates_auto.json) on NEW Stage-1 n=8 samples:
  both answers correct (math_verify), >= 2 symmetric METHOD_LEXICON terms between the two plans (best pair per question by
  sym - 5 * word Jaccard), each plan's own terms occur more often in its own execution, both executions < 5000 chars.
  Questions already used by the mini pilot (256-655) or by 2B training/validation singles are skipped.
finalize: accepted manual review (review_accept.json) + the 27 verified mini-pilot pairs (00471 excluded) -> split by question,
  near-duplicate check (word 5-gram Jaccard), bridge_dual_{train,heldout}.jsonl, normal-mode questions, split record.
encode: frozen Stage-1 layer-27 plan-text-only states -> frozen 2A step-300 compressor -> ONE latent file for every plan
  text (train, held-out, paraphrases); the 27 old pairs are checked against the 2B z_cache.
"""
import argparse
import glob
import hashlib
import itertools
import json
import os
import random
import re

key = lambda s: hashlib.sha1(s.encode()).hexdigest()
nq = lambda s: " ".join(s.lower().split())
P = "/scratch/hz33227/e2c/stage2/outputs"


def norm_plan(p):
    return re.sub(r"^I need to make a step-by-step plan\.\n", "", p)


def candidates(a):
    from later.scripts.data.plan_span_rules import METHOD_LEXICON
    used = set()
    for f in (f"{P}/mini_bridge_pilot/data/samples_stage1_n8.jsonl",):
        used |= {nq(json.loads(l)["question"]) for l in open(f)}
    for f in ("train_singles", "val_singles", "pairs"):
        used |= {nq(json.loads(l)["question"]) for l in open(f"{P}/stage2b_executor_50step/data/{f}.jsonl")}
    S = [json.loads(l) for f in a.samples.split(",") for l in open(f)]
    by = {}
    for s in S:
        if s["correct"] and nq(s["question"]) not in used:
            by.setdefault(s["record_id"], []).append(s)
    lex = sorted({m.strip() for m in METHOD_LEXICON}, key=len, reverse=True)
    terms = lambda p: {m for m in lex if re.search(r"\b" + re.escape(m), p.lower())}
    words = lambda p: set(re.findall(r"[a-z]{4,}", p.lower()))
    cnt = lambda ts, t: sum(len(re.findall(r"\b" + re.escape(m), t.lower())) for m in ts)
    out = []
    for rid, v in by.items():
        best = None
        for x, y in itertools.combinations(v, 2):
            tx, ty = terms(x["plan"]), terms(y["plan"]); wx, wy = words(x["plan"]), words(y["plan"])
            jw = len(wx & wy) / max(1, len(wx | wy)); sym = len(tx ^ ty)
            if best is None or sym - 5 * jw > best[0]:
                best = (sym - 5 * jw, sym, jw, x, y, sorted(tx - ty), sorted(ty - tx))
        if not best or best[1] < 2:
            continue
        _, sym, jw, x, y, oa, ob = best
        fa = (cnt(oa, x["execution"]), cnt(oa, y["execution"])); fb = (cnt(ob, y["execution"]), cnt(ob, x["execution"]))
        if fa[0] > fa[1] and fb[0] > fb[1] and len(x["execution"]) < 5000 and len(y["execution"]) < 5000:
            out.append({"record_id": rid, "score": best[0], "sym": sym, "word_jacc": jw, "onlyA": oa, "onlyB": ob, "sA": x["sample"], "sB": y["sample"],
                        "followA": fa, "followB": fb})
    out.sort(key=lambda c: -c["score"])
    json.dump(out, open(f"{a.out}/candidates_auto{a.tag}.json", "w"), indent=1)
    Sd = {(s["record_id"], s["sample"]): s for s in S}
    with open(f"{a.out}/review_all{a.tag}.txt", "w") as f:
        for i, c in enumerate(out):
            x, y = Sd[(c["record_id"], c["sA"])], Sd[(c["record_id"], c["sB"])]
            f.write(f"\n### [{i}] {c['record_id']} ans={x['gold_answer']!r} onlyA={c['onlyA']} onlyB={c['onlyB']} followA={c['followA']} followB={c['followB']}\n")
            f.write(f"Q: {x['question']}\n\nPLAN A:\n{norm_plan(x['plan'])}\n\nEXEC A:\n{x['execution']}\n\nPLAN B:\n{norm_plan(y['plan'])}\n\nEXEC B:\n{y['execution']}\n")
    print("questions with >=2 correct:", sum(len(v) >= 2 for v in by.values()), "| auto candidates:", len(out))


def ngr(s, n=5):
    w = re.findall(r"[a-z0-9#]+", s.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def finalize(a):
    """review_verdicts.json (per-candidate subagent verdicts) minus my_spotcheck.json rejections = accepted new pairs.
    held-out: s1val only (old heldout8 + seeded draw from the other s1val pairs); train: s1train first, then remaining s1val."""
    S = {(s["record_id"], s["sample"]): s for f in a.samples.split(",") for s in map(json.loads, open(f))}
    C = {}
    for f in glob.glob(f"{a.out}/candidates_auto*.json"):
        C.update({c["record_id"]: c for c in json.load(open(f))})
    V = json.load(open(f"{a.out}/review_verdicts.json")); my = json.load(open(f"{a.out}/my_spotcheck.json"))
    acc = {rid: m for v in V.values() for rid, m in v["accepted"].items() if rid not in my["rejected_on_my_recheck"]}
    new = []
    for rid, m in acc.items():
        c = C[rid]; x, y = S[(rid, c["sA"])], S[(rid, c["sB"])]
        assert x["correct"] and y["correct"] and x["question"] == y["question"]
        new.append({"id": rid, "question": x["question"], "answer": x["gold_answer"], "plan_A_norm": norm_plan(x["plan"]), "plan_B_norm": norm_plan(y["plan"]),
                    "execution_A": x["execution"], "execution_B": y["execution"], "method_label_A": m["label_A"], "method_label_B": m["label_B"],
                    "method_terms_only_A": c["onlyA"], "method_terms_only_B": c["onlyB"], "role": "new",
                    "source": ("s1train" if rid.startswith("s1train") else "s1val") + " Stage-1 n=8 sample, auto filter + manual review", "review_note": m.get("note", "")})
    old = [json.loads(l) for l in open(f"{P}/stage2b_executor_50step/data/pairs.jsonl")]
    old = [dict(p, source=f"s1val mini bridge pilot verified pair ({p['role']})") for p in old if not p["id"].startswith("s1val-00471")]
    allp = old + new
    assert len({nq(p["question"]) for p in allp}) == len(allp), "exact duplicate question"
    # near-duplicates: question word 5-gram Jaccard > 0.5, OR number-masked question 3-gram Jaccard > 0.4, OR number-masked plan 5-gram Jaccard > 0.5
    J = lambda x, y: len(x & y) / max(1, len(x | y))
    mask = lambda t: re.sub(r"\d+(\.\d+)?", "#", t)
    G = [(p["id"], ngr(p["question"]), ngr(mask(p["question"]), 3), ngr(mask(p["plan_A_norm"] + " " + p["plan_B_norm"]))) for p in allp]
    dups = []
    for (i, q5, q3, pl), (j, r5, r3, pm) in itertools.combinations(G, 2):
        s = {"q5": J(q5, r5), "q3_masked": J(q3, r3), "plans5_masked": J(pl, pm)}
        if s["q5"] > 0.5 or s["q3_masked"] > 0.4 or s["plans5_masked"] > 0.5:
            dups.append([i, j, {k: round(v, 3) for k, v in s.items()}])
    rng = random.Random(0)
    old_ho = [p for p in old if p["role"] == "heldout8"]
    val_rest = [p for p in allp if p["id"].startswith("s1val") and p not in old_ho]; rng.shuffle(val_rest)
    tr_pool = [p for p in new if p["id"].startswith("s1train")]; rng.shuffle(tr_pool)
    n_ho = max(0, a.n_heldout - len(old_ho))
    ho = old_ho + val_rest[:n_ho]
    tr = (tr_pool + val_rest[n_ho:])[:a.n_train]
    unused = [p["id"] for p in tr_pool + val_rest[n_ho:] if p not in tr]
    leak = [d for d in dups if ({d[0], d[1]} & {p["id"] for p in tr}) and ({d[0], d[1]} & {p["id"] for p in ho})]
    assert not leak, leak
    cols = ["id", "question", "answer", "plan_A_norm", "plan_B_norm", "execution_A", "execution_B", "method_label_A", "method_label_B",
            "method_terms_only_A", "method_terms_only_B", "source"]
    for nm, R in (("train", tr), ("heldout", ho)):
        with open(f"{a.out}/bridge_dual_{nm}.jsonl", "w") as f:
            for p in R:
                f.write(json.dumps({k: p.get(k) for k in cols}) + "\n")
    import pandas as pd
    used = {nq(p["question"]) for p in allp}
    for f in ("train_singles", "val_singles"):
        used |= {nq(json.loads(l)["question"]) for l in open(f"{P}/stage2b_executor_50step/data/{f}.jsonl")}
    df = pd.read_parquet("/scratch/hz33227/e2c/stage2/data/stage2_val.parquet").iloc[:256]
    nm_q = [r for r in df.to_dict("records") if nq(r["question"]) not in used][:16]
    with open(f"{a.out}/normal_mode_questions.jsonl", "w") as f:
        for r in nm_q:
            f.write(json.dumps({"record_id": r["record_id"], "question": r["question"], "answer": str(r["answer"])}) + "\n")
    json.dump({"n_old_reused": len(old), "n_new_accepted": len(new), "n_new_s1val": sum(p["id"].startswith("s1val") for p in new),
               "n_new_s1train": sum(p["id"].startswith("s1train") for p in new), "n_total": len(allp), "n_train": len(tr), "n_heldout": len(ho),
               "train_sources": {s: sum(p["id"].startswith(s) for p in tr) for s in ("s1train", "s1val")},
               "split_rule": "held-out = old heldout8 + random.Random(0) draw of other s1val pairs; train = s1train pairs first, then remaining s1val; held-out never s1train",
               "train_ids": [p["id"] for p in tr], "heldout_ids": [p["id"] for p in ho], "unused_ids": unused,
               "near_dup_pairs_jacc5gram_gt0.5": dups, "near_dup_train_heldout": leak,
               "excluded": {"s1val-00471": "plans A/B are the same method (earlier manual audit)", **my["rejected_on_my_recheck"]},
               "normal_mode_ids": [r["record_id"] for r in nm_q]}, open(f"{a.out}/split_record.json", "w"), indent=1)
    print("train", len(tr), "heldout", len(ho), "total", len(allp), "dups", dups)


def encode(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from later.scripts.stage2a.plan_ae import System
    texts = {}
    for f in ("bridge_dual_train.jsonl", "bridge_dual_heldout.jsonl"):
        for l in open(f"{a.out}/{f}"):
            r = json.loads(l); texts[key(r["plan_A_norm"])] = r["plan_A_norm"]; texts[key(r["plan_B_norm"])] = r["plan_B_norm"]
    for l in open(f"{a.out}/bridge_dual_paraphrase_eval.jsonl"):
        r = json.loads(l); texts[key(r["paraphrase_A"])] = r["paraphrase_A"]; texts[key(r["paraphrase_B"])] = r["paraphrase_B"]
    tok = AutoTokenizer.from_pretrained(a.model)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    st = {}
    with torch.inference_mode():
        for k, t in texts.items():
            ids = tok.encode(t, add_special_tokens=False)
            st[k] = m.model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True, use_cache=False).hidden_states[27][0].to(torch.bfloat16).cpu()
    del m; torch.cuda.empty_cache()
    R = System(a.dec, f"{a.a2}/data", seed=42); R.load(torch.load(f"{a.a2}/checkpoints/step300.pt", weights_only=False)["state"])
    shared = [k for k in st if k in R.states]
    st_diff = max([float((st[k].float() - R.states[k].float()).abs().max()) for k in shared] or [float("nan")])
    for k in st:
        R.states[k] = st[k]
    Z = {}
    with torch.no_grad():
        ks = list(texts)
        for i in range(0, len(ks), 16):
            z = R.Z([texts[k] for k in ks[i:i + 16]])
            for k, v in zip(ks[i:i + 16], z):
                Z[k] = v.float().cpu()
    zc = torch.load(f"{a.s2b}/data/z_cache.pt", weights_only=False)["Z"]
    old = [k for k in Z if k in zc]
    z_diff = max(float((Z[k] - zc[k].float()).abs().max()) for k in old)
    chk = {"n_texts": len(Z), "n_in_2A_state_cache": len(shared), "layer27_states_vs_2A_cache_maxabs": st_diff,
           "n_in_2B_zcache": len(old), "Z_vs_2B_zcache_maxabs": z_diff,
           "encoder": f"{a.a2}/checkpoints/step300.pt", "stage1": a.model, "input": "plan text only (no question / template / execution), layer 27"}
    torch.save({"Z": Z, "texts": texts, "check": chk}, f"{a.out}/stage2a_teacher_latents.pt")
    print(json.dumps(chk, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["candidates", "finalize", "encode"], required=True)
    ap.add_argument("--samples", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--n_train", type=int, default=48); ap.add_argument("--n_heldout", type=int, default=16)
    ap.add_argument("--tag", default=""); ap.add_argument("--model"); ap.add_argument("--a2"); ap.add_argument("--dec"); ap.add_argument("--s2b")
    a = ap.parse_args()
    {"candidates": candidates, "finalize": finalize, "encode": encode}[a.mode](a)
