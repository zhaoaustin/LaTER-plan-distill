"""Read-only diagnostic (no student optimizer steps): is the 2A teacher Z_T predictable from the frozen Stage-1 question
representation? Features from the untouched Stage-1 4B (LoRA B=0 at init): h_seed = final hidden at the last prompt token
(exactly what the projector sees) and mean-pooled question hidden. Closed-form ridge, lambda picked on an internal
1800/200 split of TRAIN only, reported on the 100 val singles. Also 1-NN retrieval (cosine on features)."""
import json, sys, torch, numpy as np
from later.scripts.stage2t.three_step import TStudent, kd, load_jsonl, TOK_RMS
from later.scripts.bridge.bridge_input_abcd import boot
M, S2B, O = sys.argv[1:4]
S = TStudent(M, S2B, seed=0)
tr = load_jsonl(f"{S2B}/data/train_singles.jsonl"); va = load_jsonl(f"{S2B}/data/val_singles.jsonl")
@torch.no_grad()
def feats(rows):
    H, P = [], []
    for r in rows:
        ids = torch.tensor([S.prompt_ids(r["question"])], device=S.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = S.base.model(input_ids=ids).last_hidden_state[0].float()
        H.append(hs[-1]); P.append(hs.mean(0))
    return torch.stack(H), torch.stack(P)
Htr, Ptr = feats(tr); Hva, Pva = feats(va)
Ytr = torch.stack([S.teacher(r["plan_norm"]) for r in tr]).flatten(1); Yva = torch.stack([S.teacher(r["plan_norm"]) for r in va]).flatten(1)
def kdv(P, Y):
    return ((P - Y) ** 2).mean(1) / TOK_RMS ** 2
def ridge(Xtr, Ytr, Xte, lam):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; A = (Xtr - mu) / sd; B = (Xte - mu) / sd; ym = Ytr.mean(0)
    W = torch.linalg.solve(A.T @ A + lam * torch.eye(A.size(1), device=A.device), A.T @ (Ytr - ym))
    return B @ W + ym
res = {}
ym = Ytr.mean(0, keepdim=True)
base = kdv(ym.expand_as(Yva), Yva)
res["val_kd_train_mean_baseline"] = float(base.mean())
for name, Xtr, Xva in (("h_seed", Htr, Hva), ("meanpool", Ptr, Pva)):
    grid = [1e1, 1e2, 1e3, 1e4, 1e5]
    inner = {lam: float(kdv(ridge(Xtr[:1800], Ytr[:1800], Xtr[1800:], lam), Ytr[1800:]).mean()) for lam in grid}
    lam = min(inner, key=inner.get)
    pred = ridge(Xtr, Ytr, Xva, lam)
    k = kdv(pred, Yva); ks = kdv(pred, Yva.roll(1, 0))
    gain = (base - k).tolist(); gap = (ks - k).tolist()
    Xn = torch.nn.functional.normalize(Xtr, dim=1); nn_idx = (torch.nn.functional.normalize(Xva, dim=1) @ Xn.T).argmax(1)
    knn = kdv(Ytr[nn_idx], Yva)
    res[name] = {"inner_kd_by_lambda": inner, "lambda": lam, "val_kd_ridge": float(k.mean()), "val_kd_ridge_shuffled": float(ks.mean()),
                 "gain_vs_mean_baseline": {"mean": float(np.mean(gain)), "ci95": boot(gain), "frac_pos": float(np.mean([g > 0 for g in gain]))},
                 "shuffled_minus_correct": {"mean": float(np.mean(gap)), "ci95": boot(gap), "frac_pos": float(np.mean([g > 0 for g in gap]))},
                 "rel_var_explained": float(1 - k.mean() / base.mean()), "val_kd_1nn_retrieval": float(knn.mean())}
# reference: how far apart are two plans for the SAME question vs different questions (teacher geometry)
pairs = load_jsonl(f"{S2B}/data/pairs.jsonl")
same = [float(kd(S.teacher(p["plan_A_norm"]), S.teacher(p["plan_B_norm"]))) for p in pairs]
res["teacher_kd_same_question_A_vs_B"] = float(np.mean(same))
res["teacher_kd_cross_question_val"] = float(kdv(Yva, Yva.roll(1, 0)).mean())
json.dump(res, open(f"{O}/q2z_probe.json", "w"), indent=1); print(json.dumps(res, indent=1))
