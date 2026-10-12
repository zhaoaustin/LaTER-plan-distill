"""0-step plan-paraphrase test on the 4-sample bridge-overfit step-20 executor (inference only).
Teacher latents for original and paraphrased plans go through the SAME frozen 2A pipeline: Stage-1 Qwen3-4B layer-27
states of the plan text only (cache_plan_states.py recipe) -> 2A step-300 compressor.  Injected into the 8 slots of the
unchanged executor prefix; greedy generation (same batched routine and batch composition as the overfit eval) and
teacher-forced 2x2 scores on the gold E_A / E_B."""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from later.scripts.bridge.mini_bridge import correct, last_boxed, method_hits, ngram_overlap
from later.scripts.stage2s.stochastic_pilot import key, short
from later.scripts.stage2t.bridge_oracle import scores
from later.scripts.stage2t.contrastive_2b import generate_batch, train_pairs
from later.scripts.stage2t.three_step import TStudent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--s2b", required=True); ap.add_argument("--a2", required=True); ap.add_argument("--dec", required=True)
    ap.add_argument("--init", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--para", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gen_tokens", type=int, default=2048)
    a = ap.parse_args(); torch.manual_seed(0)
    para = json.load(open(a.para)); ids = list(para)
    S = TStudent(a.model, a.s2b, seed=0); S.load(torch.load(a.init, weights_only=False)["state"])
    ck = torch.load(a.ckpt, weights_only=False); S.load(ck["state"]); S.peft.eval()
    for p_ in S.peft.parameters():
        p_.requires_grad_(False)
    tr, _, _ = train_pairs(S, a.s2b); P = [next(p for p in tr if short(p["id"]) == i) for i in ids]
    res = {"executor_init": a.init, "executor_ckpt": a.ckpt, "ckpt_step": ck.get("step"), "encoder": f"{a.a2}/checkpoints/step300.pt", "seed": 0}
    # ---- layer-27 plan states with the clean Stage-1 model (same recipe as cache_plan_states.py)
    texts = {}
    for p, i in zip(P, ids):
        for X in ("A", "B"):
            texts[(i, X)] = p[f"plan_{X}_norm"]; texts[(i, X + "'")] = para[i][X]
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    st = {}
    with torch.inference_mode():
        for k, t in texts.items():
            tid = S.tok.encode(t, add_special_tokens=False)
            st[k] = m.model(input_ids=torch.tensor([tid], device="cuda"), output_hidden_states=True, use_cache=False).hidden_states[27][0].to(torch.bfloat16).cpu()
    del m; torch.cuda.empty_cache()
    from later.scripts.stage2a.plan_ae import System
    R = System(a.dec, f"{a.a2}/data", seed=42); R.load(torch.load(f"{a.a2}/checkpoints/step300.pt", weights_only=False)["state"])
    res["orig_states_vs_2A_cache_maxabs"] = max(float((st[(i, X)].float() - R.states[key(texts[(i, X)])].float()).abs().max()) for i in ids for X in ("A", "B"))
    for k, t in texts.items():
        R.states[key(t)] = st[k]
    with torch.no_grad():
        Z = {k: R.Z([t])[0] for k, t in texts.items()}
    res["orig_Z_vs_2B_zcache_maxabs"] = max(float((Z[(i, X)] - S.Zraw[key(texts[(i, X)])].float().to(Z[(i, X)].device)).abs().max()) for i in ids for X in ("A", "B"))
    del R; torch.cuda.empty_cache()
    # ---- cosines
    cs = lambda x, y: float(F.cosine_similarity(x.flatten(), y.flatten(), 0))
    sl = lambda x, y: [round(float(v), 4) for v in F.cosine_similarity(x, y, -1)]
    res["cos"] = {i: {"A_A'": cs(Z[(i, "A")], Z[(i, "A'")]), "B_B'": cs(Z[(i, "B")], Z[(i, "B'")]), "A_B'": cs(Z[(i, "A")], Z[(i, "B'")]), "B_A'": cs(Z[(i, "B")], Z[(i, "A'")]),
                      "A_B": cs(Z[(i, "A")], Z[(i, "B")]), "A'_B'": cs(Z[(i, "A'")], Z[(i, "B'")]),
                      "slot_A_A'": sl(Z[(i, "A")], Z[(i, "A'")]), "slot_B_B'": sl(Z[(i, "B")], Z[(i, "B'")]), "slot_A_B'": sl(Z[(i, "A")], Z[(i, "B'")]),
                      "slot_B_A'": sl(Z[(i, "B")], Z[(i, "A'")]), "slot_A_B": sl(Z[(i, "A")], Z[(i, "B")]),
                      "norms": {k[1]: float(v.norm()) for k, v in Z.items() if k[0] == i}} for i in ids}
    # ---- 2x2 scores (original and paraphrase latents) on the gold E_A / E_B
    res["scores"] = {}
    with torch.no_grad():
        for p, i in zip(P, ids):
            res["scores"][i] = {"orig": scores(S, p)}
            pp = dict(p); pp["plan_A_norm"], pp["plan_B_norm"] = para[i]["A"], para[i]["B"]
            S.Zraw[key(para[i]["A"])], S.Zraw[key(para[i]["B"])] = Z[(i, "A'")].cpu(), Z[(i, "B'")].cpu()
            res["scores"][i]["para"] = scores(S, pp)
    # ---- generation: 2 questions x [Z_A, Z_B, Z_A', Z_B'] in one batch (fresh KV for the batch)
    conds = ["A", "B", "A'", "B'"]
    items = [(p, i, c) for p, i in zip(P, ids) for c in conds]
    gens = generate_batch(S, [p["question"] for p, _, _ in items], torch.stack([Z[(i, c)].to(S.dev) for _, i, c in items]), a.gen_tokens)
    res["gens"] = []
    for (p, i, c), g in zip(items, gens):
        pr = last_boxed(g); oa, ob = ngram_overlap(g, p["execution_A"]), ngram_overlap(g, p["execution_B"])
        res["gens"].append({"id": i, "latent": "Z_" + c, "target_strategy": c[0], "auto_label": "A" if oa > ob else ("B" if ob > oa else "="), "ov_A": oa, "ov_B": ob,
                            "terms_A": method_hits(g, p["method_terms_only_A"]), "terms_B": method_hits(g, p["method_terms_only_B"]),
                            "pred": pr, "correct": correct(pr, str(p["answer"])), "format_ok": "</EXECUTION>" in g and "<|im_end|>" in g,
                            "identical_to_orig_same_side": None, "text": g})
    for r in res["gens"]:
        if r["latent"].endswith("'"):
            o = next(x for x in res["gens"] if x["id"] == r["id"] and x["latent"] == r["latent"][:-1])
            r["identical_to_orig_same_side"] = r["text"] == o["text"]
    res["texts"] = {f"{i}_{c}": t for (i, c), t in texts.items()}
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(f"{a.out}/results.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("gens", "texts")}, indent=1))
    for r in res["gens"]:
        print(r["id"], r["latent"], r["auto_label"], r["correct"], r["format_ok"], r["identical_to_orig_same_side"])


if __name__ == "__main__":
    main()
