import json, sys, torch
from later.scripts.stage2a.plan_ae import System
from later.scripts.bridge.mini_bridge import ngram_overlap, method_hits
A2, S2B, O = sys.argv[1:4]
S = System("/home/hz33227/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca", f"{A2}/data", seed=42)
S.load(torch.load(f"{A2}/checkpoints/step300.pt", weights_only=False)["state"]); S.peft.eval(); S.comp.eval()
pairs = [json.loads(l) for l in open(f"{S2B}/data/pairs.jsonl")]
rows = []
for p in pairs:
    with torch.no_grad():
        Z = S.Z([p["plan_A_norm"], p["plan_B_norm"]])
        out = {}
        for X, z in (("A", Z[0]), ("B", Z[1])):
            t, _ = S.generate(z, 320)
            out[X] = {"text": t, "ov_A": ngram_overlap(t, p["plan_A_norm"]), "ov_B": ngram_overlap(t, p["plan_B_norm"]),
                      "terms_A": method_hits(t, p["method_terms_only_A"]), "terms_B": method_hits(t, p["method_terms_only_B"])}
    rows.append({"id": p["id"], "role": p["role"], "method_A": p["method_label_A"], "method_B": p["method_label_B"], "gen": out})
    print(p["id"], p["role"], {X: (round(v["ov_A"], 3), round(v["ov_B"], 3), v["terms_A"], v["terms_B"]) for X, v in out.items()}, flush=True)
json.dump(rows, open(f"{O}/teacher_reader_freerun.json", "w"), indent=1)
