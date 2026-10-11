"""Stage 2A: cache frozen Stage-1 Qwen3-4B layer-27 per-token hidden states of PLAN TEXT ONLY (no question, no chat
template, no execution). Key = sha1(plan text). Also fits train-only per-dim standardization stats."""
import argparse, hashlib, json, os, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--layer", type=int, default=27)
a = ap.parse_args()
key = lambda s: hashlib.sha1(s.encode()).hexdigest()
texts = {}
for split in ("train", "val"):
    for l in open(f"{a.data}/{split}_plans.jsonl"):
        r = json.loads(l); texts[key(r["plan_norm"])] = (r["plan_norm"], split)
for l in open(f"{a.data}/paired_plans.jsonl"):
    r = json.loads(l)
    for k in ("plan_A_norm", "plan_B_norm"):
        texts.setdefault(key(r[k]), (r[k], "paired"))
for l in open(f"{a.data}/paraphrases.jsonl"):
    r = json.loads(l); texts.setdefault(key(r["paraphrase"]), (r["paraphrase"], "paraphrase"))
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
assert model.config.num_hidden_layers >= a.layer
fp = lambda: float(sum(p.double().sum() for p in model.parameters()))
f0 = fp(); t0 = time.time()
cache, ntok = {}, {}
with torch.inference_mode():
    for k, (t, split) in texts.items():
        ids = tok.encode(t, add_special_tokens=False)
        h = model.model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True, use_cache=False).hidden_states[a.layer][0]
        cache[k] = h.to(torch.bfloat16).cpu(); ntok[k] = len(ids)
f1 = fp()
X = torch.cat([cache[k].float() for k, (t, s) in texts.items() if s == "train"])
mu, sd = X.mean(0), X.std(0)
floor = float(np.percentile(sd[sd > 0].numpy(), 5.0))
torch.save({"states": cache, "layer": a.layer, "model": a.model, "dtype": "bfloat16",
            "input": "plan_norm text only, tokenizer.encode(add_special_tokens=False), no question / chat template / execution"},
           f"{a.data}/plan_states_L{a.layer}.pt")
np.savez(f"{a.data}/train_stats_L{a.layer}.npz", mu=mu.numpy(), sigma=sd.numpy(), floor=floor, clip=5.0, floor_pct=5.0, n_tokens=X.size(0))
json.dump({"n_texts": len(cache), "by_split": {s: sum(1 for v in texts.values() if v[1] == s) for s in ("train", "val", "paired", "paraphrase")},
           "layer": a.layer, "hidden": int(X.size(1)), "train_tokens": int(X.size(0)), "sigma_floor": floor,
           "frac_dims_floored": float((sd < floor).float().mean()), "model_param_sum_before": f0, "model_param_sum_after": f1,
           "model_unchanged": f0 == f1, "seconds": time.time() - t0,
           "mean_plan_tokens_encoder": float(np.mean(list(ntok.values())))}, open(f"{a.data}/cache_manifest.json", "w"), indent=1)
print(open(f"{a.data}/cache_manifest.json").read())
