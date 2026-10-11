"""Stage 2B data + frozen Stage-2A Z cache.
- Loads the Stage-2A system (frozen 4B plan-state cache + compressor step 300; the 0.6B decoder only to re-verify
  the step-300 paired margin), computes Z = compressor(H(P)) [8, 2560] for every plan used in 2B, caches Z only.
- Train singles: 2000 of the 2A train plans (seed 42), joined to stage2_train by record_id for Q / E / A.
  Val singles: 100 of the 2A val plans joined to stage2_val. Plans containing the final answer string are filtered.
- Pairs: 20 bridge pairs (train, with ranking) + 8 held-out pairs (eval only) with Q, P_A/E_A, P_B/E_B.
"""
import argparse, hashlib, json, os, re
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer
from later.scripts.stage2a.plan_ae import System, SpanTagger, paired, load_jsonl, key
from later.src.train.stage2_plan_dataset import EOS, EXEC_CLOSE

ap = argparse.ArgumentParser()
ap.add_argument("--a2", required=True); ap.add_argument("--dec", required=True); ap.add_argument("--rules", required=True)
ap.add_argument("--data_dir", required=True); ap.add_argument("--bridge", required=True); ap.add_argument("--exec_tok", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--n_train", type=int, default=2000); ap.add_argument("--n_val", type=int, default=100)
ap.add_argument("--max_exec_tokens", type=int, default=3072)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
nq = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()
norm = lambda s: re.sub(r"\s+", "", str(s)).lower()
S = System(a.dec, f"{a.a2}/data", seed=42)
ck = torch.load(f"{a.a2}/checkpoints/step300.pt", weights_only=False)
S.load(ck["state"]); S.comp.eval(); S.adapter.eval(); S.peft.eval()
tagger = SpanTagger(S.tok, a.rules)
pairs_2a = load_jsonl(f"{a.a2}/data/paired_plans.jsonl")
with torch.no_grad():
    pr, _ = paired(S, pairs_2a, tagger)
verify = {"stage2a_step300_heldout8_paired_delta_recomputed": pr["heldout8"]["mean_delta"], "reported": 0.10978858545422554,
          "abs_diff": abs(pr["heldout8"]["mean_delta"] - 0.10978858545422554)}
print("VERIFY", verify, flush=True)
etok = AutoTokenizer.from_pretrained(a.exec_tok)
tr = pd.read_parquet(f"{a.data_dir}/stage2_train.parquet").set_index("record_id")
va = pd.read_parquet(f"{a.data_dir}/stage2_val.parquet").set_index("record_id")
def singles(path, df, n, split):
    rows = load_jsonl(path); stats = {"candidates": len(rows), "answer_in_plan": 0, "exec_too_long": 0}
    out = []
    for r in rows:
        x = df.loc[r["record_id"]]
        ans = norm(x["answer"])
        if len(ans) >= 2 and ans in norm(r["plan_norm"]):
            stats["answer_in_plan"] += 1; continue
        nt = len(etok.encode(x["execution"] + EXEC_CLOSE + EOS, add_special_tokens=False))
        if nt > a.max_exec_tokens:
            stats["exec_too_long"] += 1; continue
        out.append({"id": r["record_id"], "split": split, "question": x["question"], "plan_norm": r["plan_norm"], "plan_hash": key(r["plan_norm"]),
                    "execution": x["execution"], "answer": x["answer"], "exec_tokens": nt})
    rng = np.random.default_rng(42); idx = sorted(rng.choice(len(out), size=min(n, len(out)), replace=False))
    sel = [out[i] for i in idx]
    L = np.array([s["exec_tokens"] for s in out])
    stats.update({"eligible": len(out), "selected": len(sel), "exec_tokens_p50_p95_max_eligible": [float(np.percentile(L, 50)), float(np.percentile(L, 95)), int(L.max())]})
    return sel, stats
tr_s, tr_stats = singles(f"{a.a2}/data/train_plans.jsonl", tr, a.n_train, "train")
va_s, va_stats = singles(f"{a.a2}/data/val_plans.jsonl", va, a.n_val, "val")
br = {p["id"]: p for f in ("train_pairs.jsonl", "heldout_pairs.jsonl") for p in load_jsonl(f"{a.bridge}/{f}")}
pairs = []
for p in pairs_2a:
    b = br[p["id"]]
    pairs.append({"id": p["id"], "role": "train_pair" if p["role"] == "extended20" else "heldout8", "question": b["question"], "answer": b["answer"],
                  "plan_A_norm": p["plan_A_norm"], "plan_B_norm": p["plan_B_norm"], "execution_A": b["execution_A"], "execution_B": b["execution_B"],
                  "method_label_A": p["method_label_A"], "method_label_B": p["method_label_B"],
                  "method_terms_only_A": p["method_terms_only_A"], "method_terms_only_B": p["method_terms_only_B"]})
# question-level disjointness
qs = lambda L: {nq(x["question"]) for x in L}
held_q = qs([p for p in pairs if p["role"] == "heldout8"]); trp_q = qs([p for p in pairs if p["role"] == "train_pair"])
assert not held_q & qs(tr_s) and not held_q & qs(va_s) and not held_q & trp_q and not qs(tr_s) & qs(va_s)
# Z cache (frozen compressor, no grad)
texts = {s["plan_norm"] for s in tr_s + va_s} | {p[k] for p in pairs for k in ("plan_A_norm", "plan_B_norm")}
texts = sorted(texts); Z = {}
with torch.no_grad():
    for i in range(0, len(texts), 32):
        zb = S.Z(texts[i:i + 32])
        for t, z in zip(texts[i:i + 32], zb):
            Z[key(t)] = z.cpu()
    z_online = S.Z(texts[:4]).cpu()
online_vs_cache = float(max((z_online[j] - Z[key(texts[j])]).abs().max() for j in range(4)))
Zm = torch.stack([Z[s["plan_hash"]] for s in tr_s]).mean(0)
Zm = Zm * (0.02195320464670658 / Zm.pow(2).mean(-1, keepdim=True).sqrt())
torch.save({"Z": Z, "Z_mean_train": Zm, "stage2a_ckpt": f"{a.a2}/checkpoints/step300.pt", "dim": int(Zm.size(-1)), "slots": int(Zm.size(0))}, f"{a.out}/z_cache.pt")
for name, L in (("train_singles", tr_s), ("val_singles", va_s), ("pairs", pairs)):
    with open(f"{a.out}/{name}.jsonl", "w") as f:
        for r in L:
            f.write(json.dumps(r) + "\n")
ck_sha = hashlib.sha256(open(f"{a.a2}/checkpoints/step300.pt", "rb").read()).hexdigest()
man = {"verify": verify, "stage2a_step300_sha256": ck_sha, "train_singles": tr_stats, "val_singles": va_stats,
       "n_pairs": {"train_pair": len(trp_q), "heldout8": len(held_q)}, "n_Z": len(Z), "z_shape": [int(Zm.size(0)), int(Zm.size(1))],
       "z_rms_mean": float(torch.stack(list(Z.values())).pow(2).mean(-1).sqrt().mean()), "online_vs_cache_max_abs": online_vs_cache,
       "question_disjoint": True, "encoder_input": "plan_norm text only (Stage-2A cache, layer 27, train-only standardization)",
       "decoder_0.6B_adapter_used_for_Z": False}
json.dump(man, open(f"{a.out}/stage2a_latent_cache_manifest.json", "w"), indent=1)
print(json.dumps(man, indent=1))
