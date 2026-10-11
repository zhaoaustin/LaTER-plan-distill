"""Step-1 control: are the two plans of a question separable in the bridge INPUT (frozen encoder states)?
cos(A, B) of mean-pooled plan states at several layers: raw, LayerNorm'd (as the bridge sees them), and with the
largest-|mean| outlier dimensions removed; plus the same for plans of DIFFERENT questions as a reference."""
import json, os, sys, torch, torch.nn.functional as F
from later.scripts.bridge.mini_bridge import Pilot, load_pairs
import later.scripts.bridge.mini_bridge as mb
P = Pilot(sys.argv[1]); out = sys.argv[2]
pairs = load_pairs(os.path.join(out, "train_pairs.jsonl")) + load_pairs(os.path.join(out, "heldout_pairs.jsonl"))
res = {}
for layer in (9, 18, 27, 36):
    mb.ENC_LAYER = layer; P.cache = {}
    MA, MB, LA, LB = [], [], [], []
    for p in pairs:
        ha, hb = P.plan_states(p["question"], p["plan_A"]).float(), P.plan_states(p["question"], p["plan_B"]).float()
        MA.append(ha.mean(0)); MB.append(hb.mean(0))
        LA.append(F.layer_norm(ha, ha.shape[-1:]).mean(0)); LB.append(F.layer_norm(hb, hb.shape[-1:]).mean(0))
    MA, MB, LA, LB = map(torch.stack, (MA, MB, LA, LB))
    allm = torch.cat([MA, MB]); out_dims = allm.mean(0).abs().topk(8).indices
    keep = torch.ones(MA.size(1), dtype=torch.bool); keep[out_dims] = False
    c = lambda X, Y: float(F.cosine_similarity(X, Y, dim=-1).mean())
    rolled = lambda X: torch.roll(X, 1, 0)
    mu = allm.mean(0)
    res[layer] = {"same_q_raw": c(MA, MB), "diff_q_raw": c(MA, rolled(MB)),
                  "same_q_layernorm": c(LA, LB), "diff_q_layernorm": c(LA, rolled(LB)),
                  "same_q_no_outlier_dims": c(MA[:, keep], MB[:, keep]), "diff_q_no_outlier_dims": c(MA[:, keep], rolled(MB)[:, keep]),
                  "same_q_centered": c(MA - mu, MB - mu), "diff_q_centered": c(MA - mu, rolled(MB) - mu),
                  "top_outlier_dim_share_of_norm": float(allm[:, out_dims].norm(dim=-1).mean() / allm.norm(dim=-1).mean())}
    print(layer, {k: round(v, 4) for k, v in res[layer].items()}, flush=True)
json.dump(res, open(os.path.join(out, "step1_rank_weight", "encoder_state_check.json"), "w"), indent=1)
