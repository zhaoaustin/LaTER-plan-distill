"""Unit / smoke tests for contrastive_2b (needs 1 GPU + the joint step-50 checkpoint).
python -m later.scripts.stage2t.test_contrastive --model M --s2b S2B --ckpt joint_step50.pt --out OUT  -> OUT/test_contrastive.json"""
import argparse
import json
import time

import torch
import torch.nn.functional as F

from later.scripts.stage2s.stochastic_pilot import grad_norm, key
from later.scripts.stage2t import contrastive_2b as C
from later.scripts.stage2t.three_step import D, TStudent, load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--s2b", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = {}
    # ---- pure-tensor tests
    torch.manual_seed(0)
    lg = torch.randn(6, 1000) * 3
    r["jsd_pp"] = float(C.token_jsd(lg, lg).abs().max())
    js_small, js_big = C.token_jsd(lg, lg + 0.1 * torch.randn_like(lg)).mean(), C.token_jsd(lg, lg + 3.0 * torch.randn_like(lg)).mean()
    r["jsd_monotone"] = [float(js_small), float(js_big)]
    r["jsd_nonneg"] = bool((C.token_jsd(lg, torch.randn_like(lg) * 5) >= -1e-7).all())
    r["jsd_le_ln2"] = bool((C.token_jsd(lg * 10, -lg * 10) <= 0.6932).all())
    la = lg.clone().requires_grad_(True); lb = (lg + torch.randn_like(lg)).requires_grad_(True)
    loss = -C.token_jsd(la, lb).mean(); loss.backward()
    r["blind_gets_no_grad"] = lb.grad is None; r["aware_gets_grad"] = la.grad is not None and float(la.grad.norm()) > 0
    with torch.no_grad():
        la2 = la - 50.0 * la.grad                                  # one descent step on L_anchor = -JSD
    r["descent_on_neg_jsd_increases_jsd"] = [float(C.token_jsd(lg, lb.detach()).mean()), float(C.token_jsd(la2, lb.detach()).mean())]
    d = torch.tensor([0.5, -0.5], requires_grad=True); lr_ = 0.5 * (F.softplus(-d[0]) + F.softplus(-d[1])); lr_.backward()
    r["rank_grad_pushes_gaps_up"] = bool((d.grad < 0).all())
    # ---- model tests
    S = TStudent(a.model, a.s2b, seed=0)
    C.N_TAIL = len(S.tok.encode("</EXECUTION><|im_end|>", add_special_tokens=False))
    ck = torch.load(a.ckpt, weights_only=False); S.load(ck["state"])
    singles = load_jsonl(f"{a.s2b}/data/train_singles.jsonl")
    trainp, held, allp = C.train_pairs(S, a.s2b)
    params = S.params(); s0 = singles[0]; p0 = trainp[0]
    zraw_snap = {k: v.clone() for k, v in list(S.Zraw.items())[:50]}
    # anchor gradient routing (anchor term alone)
    for p in params:
        p.grad = None
    q = s0["question"]; tgt = S.target_ids(s0["execution"]); nT = C.n_anchor(tgt)
    Zg = S.latents(q, torch.zeros(1, D, device=S.dev), 0.0, 1.0); Zg.retain_grad()
    _, _, lga, _ = C.exec_forward(S, q, Zg, [tgt], n_logits=nT)
    with torch.no_grad():
        _, _, lgb, _ = C.exec_forward(S, q, C.blind_latent(Zg[0])[None], [tgt], n_logits=nT)
    r["blind_logits_requires_grad"] = bool(lgb[0].requires_grad)
    (-C.token_jsd(lga[0], lgb[0]).mean()).backward()
    r["anchor_grad"] = {"projector": grad_norm([p for _, p in S.proj]), "lora": grad_norm([p for _, p in S.lora]),
                        "dZ_per_slot": [float(x) for x in Zg.grad[0].norm(dim=-1)]}
    r["anchor_grad_ok"] = r["anchor_grad"]["projector"] > 0 and r["anchor_grad"]["lora"] > 0 and r["anchor_grad"]["dZ_per_slot"][0] > 0
    # full single loss (A) gradients finite
    for p in params:
        p.grad = None
    loss, parts, _ = C.single_loss(S, s0, 0.01); loss.backward()
    r["A_single_parts"] = {k: float(v) for k, v in parts.items()}
    r["A_grads_finite"] = all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)
    # plan term routing: only LoRA (teacher latents frozen)
    for p in params:
        p.grad = None
    lp, pp = C.plan_terms(S, p0, 1.0, 1.0); lp.backward()
    r["plan_parts"] = {k: float(v) for k, v in pp.items()}
    r["plan_grad"] = {"projector": grad_norm([p for _, p in S.proj]), "lora": grad_norm([p for _, p in S.lora])}
    r["plan_grad_ok"] = r["plan_grad"]["projector"] == 0 and r["plan_grad"]["lora"] > 0
    # A/B swap flips the rank constraint: swapped pair's dA == original dB
    sw = dict(p0)
    for k in ("plan_A_norm", "tgt_A"):
        k2 = k.replace("_A", "_B"); sw[k], sw[k2] = p0[k2], p0[k]
    with torch.no_grad():
        _, p1 = C.plan_terms(S, p0, 1.0, 1.0); _, p2 = C.plan_terms(S, sw, 1.0, 1.0)
        same = dict(p0); same["plan_B_norm"] = p0["plan_A_norm"]; _, p3 = C.plan_terms(S, same, 1.0, 1.0)
    r["swap"] = {"orig_dA_dB": [float(p1["dA"]), float(p1["dB"])], "swapped_dA_dB": [float(p2["dA"]), float(p2["dB"])],
                 "same_Z_dA_dB": [float(p3["dA"]), float(p3["dB"])]}
    r["swap_ok"] = abs(float(p1["dA"]) - float(p2["dB"])) < 2e-3 and abs(float(p1["dB"]) - float(p2["dA"])) < 2e-3 and \
        abs(float(p3["dA"])) < 1e-6 and abs(float(p3["dB"])) < 1e-6
    r["strategy_span_example"] = {"id": p0["id"], "span_A": C.strategy_span(p0, "A"), "len_A": len(p0["tgt_A"]),
                                  "span_A_head": S.tok.decode(p0["tgt_A"][C.strategy_span(p0, "A")[0]:C.strategy_span(p0, "A")[0] + 20]),
                                  "span_A_tail": S.tok.decode(p0["tgt_A"][C.strategy_span(p0, "A")[1] - 10:C.strategy_span(p0, "A")[1]])}
    # masks / no leakage: prompt holds neither plan nor execution; targets do not hold the plan
    dec = S.tok.decode(S.prompt_ids(p0["question"]))
    r["no_plan_exec_in_prompt"] = all(t[10:80] not in dec for t in (p0["plan_A_norm"], p0["plan_B_norm"], p0["execution_A"], p0["execution_B"], s0["plan_norm"]))
    # one optimizer step for each branch from the checkpoint: finite, base frozen, timing / memory
    snap = {n: p.detach().clone() for n, p in S.peft.named_parameters() if not p.requires_grad}
    res = {}
    for tag, la_, lp_ in (("D0", 0.0, 0.0), ("A", 0.01, 0.0), ("B", 0.0, 0.02)):
        S.load(ck["state"]); opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0); opt.load_state_dict(ck["opt"])
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time(); opt.zero_grad(set_to_none=True)
        for s in singles[:4]:
            l, _, _ = C.single_loss(S, s, la_); (l / 4).backward()
        if lp_ > 0:
            for p in trainp[:2]:
                l, _ = C.plan_terms(S, p, 1.0, 1.0); (lp_ * l / 2).backward()
        fin = all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0)); opt.step(); torch.cuda.synchronize()
        res[tag] = {"finite": fin, "grad_norm": gn, "sec_4singles(+2pairs)": time.time() - t0, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
    r["one_step"] = res
    r["base_unchanged"] = all(torch.equal(p, snap[n]) for n, p in S.peft.named_parameters() if not p.requires_grad)
    r["teacher_cache_unchanged"] = all(torch.equal(S.Zraw[k], v) for k, v in zraw_snap.items())
    hard = {"jsd_pp_zero": r["jsd_pp"] < 1e-6, "jsd_monotone": r["jsd_monotone"][0] < r["jsd_monotone"][1], "jsd_nonneg": r["jsd_nonneg"], "jsd_le_ln2": r["jsd_le_ln2"],
            "blind_no_grad": r["blind_gets_no_grad"] and not r["blind_logits_requires_grad"], "aware_grad": r["aware_gets_grad"],
            "neg_jsd_sign": r["descent_on_neg_jsd_increases_jsd"][1] > r["descent_on_neg_jsd_increases_jsd"][0], "rank_sign": r["rank_grad_pushes_gaps_up"],
            "anchor_routing": r["anchor_grad_ok"], "A_finite": r["A_grads_finite"], "plan_routing_lora_only": r["plan_grad_ok"], "ab_swap": r["swap_ok"],
            "no_leak": r["no_plan_exec_in_prompt"], "one_step_finite": all(v["finite"] for v in res.values()), "base_frozen": r["base_unchanged"],
            "encoder_cache_frozen": r["teacher_cache_unchanged"]}
    r["hard"] = hard; r["ALL_PASS"] = all(hard.values())
    json.dump(r, open(f"{a.out}/test_contrastive.json", "w"), indent=1); print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
