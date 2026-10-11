"""Step-1 control: can ANY latent steer the frozen Stage-1 executor? No bridge: per-pair free parameters Z_A, Z_B
[8, d] (each vector projected to the token-embedding RMS after every step), optimised on that pair only with
L = CE + lambda * rank (same definitions as mini_bridge). Reports the margin reached vs the text-plan oracle (0.167).
This measures steerability/capacity of the embedding-slot interface, not generalisation."""
import argparse, json, os, random
import torch, torch.nn.functional as F
from later.scripts.bridge.mini_bridge import Pilot, load_pairs, L_SLOTS

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); ap.add_argument("--run_dir", required=True)
ap.add_argument("--n_pairs", type=int, default=8); ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--lambda_rank", type=float, default=1.0)
a = ap.parse_args()
os.makedirs(a.run_dir, exist_ok=True)
torch.manual_seed(0)
P = Pilot(a.model)
d = P.model.config.hidden_size
pairs = load_pairs(os.path.join(a.out, "train_pairs.jsonl"))[: a.n_pairs]
proj = lambda z: z * (P.tok_rms / z.pow(2).mean(-1, keepdim=True).add(1e-12).sqrt())
res = []
log = open(os.path.join(a.run_dir, "log.jsonl"), "w")
for p in pairs:
    Z = torch.nn.Parameter(proj(torch.randn(2, L_SLOTS, d, device=P.dev)))
    opt = torch.optim.Adam([Z], lr=a.lr * P.tok_rms / 0.02)
    hist = []
    for step in range(a.steps + 1):
        s, _ = P.exec_scores(p["question"], torch.stack([Z[0], Z[1], Z[1], Z[0]]),
                             [p["execution_A"], p["execution_B"], p["execution_A"], p["execution_B"]])
        mA, mB = s[0] - s[2], s[1] - s[3]
        l_ce = -(s[0] + s[1]) / 2
        loss = l_ce + a.lambda_rank * (F.softplus(-mA) + F.softplus(-mB))
        hist.append({"step": step, "margin_A": float(mA), "margin_B": float(mB), "ce": float(l_ce),
                     "z_cos_AB": float(F.cosine_similarity(Z[0], Z[1], dim=-1).mean())})
        if step == a.steps:
            break
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            Z.copy_(proj(Z))
    r = {"id": p["id"], "start": hist[0], "end": hist[-1], "best_mean_margin": max((h["margin_A"] + h["margin_B"]) / 2 for h in hist)}
    res.append(r); log.write(json.dumps({"id": p["id"], "hist": hist}) + "\n"); log.flush()
    print(p["id"], "margin A/B %.4f/%.4f -> %.4f/%.4f  ce %.3f -> %.3f  zcos %.3f -> %.3f" % (
        r["start"]["margin_A"], r["start"]["margin_B"], r["end"]["margin_A"], r["end"]["margin_B"],
        r["start"]["ce"], r["end"]["ce"], r["start"]["z_cos_AB"], r["end"]["z_cos_AB"]), flush=True)
summ = {"mean_end_margin": sum((r["end"]["margin_A"] + r["end"]["margin_B"]) / 2 for r in res) / len(res),
        "frac_end_margins_positive": sum((r["end"]["margin_A"] > 0) + (r["end"]["margin_B"] > 0) for r in res) / (2 * len(res)),
        "mean_end_ce": sum(r["end"]["ce"] for r in res) / len(res), "mean_start_ce": sum(r["start"]["ce"] for r in res) / len(res),
        "oracle_text_plan_margin_heldout": 0.1673, "config": vars(a), "pairs": res}
json.dump(summ, open(os.path.join(a.run_dir, "summary.json"), "w"), indent=1)
print({k: v for k, v in summ.items() if k not in ("pairs", "config")})
