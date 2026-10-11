"""Supplement to P0 (forward-only): KL noise floor. Same question, correct latents only, first 16 execution targets:
KL(alpha=1 || alpha=1.001)  -> bf16 numerical floor for a negligible latent change
KL(alpha=1 || alpha=2 / 4)  -> effect of amplifying the question's OWN residual
Also the full-latent condition KL(learned || zero latents) as a large-perturbation reference. Eval = first N val records."""
import json, os, sys
import numpy as np, pandas as pd, torch
from later.scripts.train.diag_latent_step_collapse import builder, exec_forward
from later.src.train.stage2_common import load_stage2_model, load_tokenizer
from later.src.train.stage2_plan_dataset import Stage2Collator, build_views, stage2_tokens

model_path, data_dir, out, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tok = load_tokenizer(model_path); T = stage2_tokens(tok)
val = pd.read_parquet(f"{data_dir}/stage2_val.parquet")
rows = [val.iloc[i].to_dict() for i in range(N)]
model = load_stage2_model(model_path, latent_embed_builder_no_grad=True, latent_projector_dropout=0.0, dtype=torch.float32).cuda().eval()
mu = torch.load(os.path.join(out, "latent_mu.pt"))["mu"].cuda()
col = Stage2Collator(tok.pad_token_id)
conds = {"alpha1.001": lambda Z: mu + 1.001 * (Z - mu), "alpha2": lambda Z: mu + 2 * (Z - mu),
         "alpha4": lambda Z: mu + 4 * (Z - mu), "zero_latents": lambda Z: torch.zeros_like(Z),
         "mu_only(alpha0)": lambda Z: mu.expand_as(Z).clone()}
kl = {c: [] for c in conds}
with torch.inference_mode():
    for s in range(0, N, 8):
        b = col([build_views(tok, r, 8, T, 4096) for r in rows[s:s + 8]])
        Z, _ = builder(model, b)
        base = exec_forward(model, b, latent_override=Z)
        cnt = base["target_mask"].sum(1).tolist()
        lb = torch.log_softmax(base["target_logits"].float(), -1)
        for c, f in conds.items():
            o = exec_forward(model, b, latent_override=f(Z))
            lo = torch.log_softmax(o["target_logits"].float(), -1)
            k = 0
            for n in cnt:
                p, q = lb[k:k + min(16, n)], lo[k:k + min(16, n)]
                kl[c].append(float((p.exp() * (p - q)).sum(-1).mean())); k += n
res = {c: {"mean": float(np.mean(v)), "median": float(np.median(v)), "max": float(np.max(v)), "n": len(v)} for c, v in kl.items()}
res["_def"] = "KL(correct alpha=1 || condition), mean over first 16 execution targets, then over questions"
json.dump(res, open(os.path.join(out, "kl_noise_floor.json"), "w"), indent=1)
print(json.dumps(res))
