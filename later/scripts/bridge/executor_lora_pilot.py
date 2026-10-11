"""Executor-LoRA conditioning pilot: FROZEN B-50 Bridge (standardized input) + Stage-1 executor with a trainable LoRA.

Single new variable vs the A/B/C/D B group: the executor may adapt through LoRA (r=16, alpha=32, dropout 0, q/v/o_proj).
- Z_A / Z_B for all 28 pairs are computed ONCE with the frozen Stage-1 model (no LoRA attached yet) + frozen B-50
  bridge + the saved train-only standardization stats, and cached. After LoRA is attached they are re-computed online
  with the adapter DISABLED to verify bit-identity -> the bridge input never sees LoRA.
- Loss = the B-group implementation unchanged: 1/2 [CE_full(E_A|Z_A) + CE_full(E_B|Z_B)] + 0.1 [softplus(-D_A) +
  softplus(-D_B)], D on the first 64 execution targets (same positions / mask for both Z).
- Same data, data order (seed 1234), pairs/step 4, AdamW lr 1e-4 wd 0.01, grad clip 1.0, 50 steps.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

import later.scripts.bridge.mini_bridge as mb
from later.scripts.bridge.bridge_input_abcd import PilotABCD, boot
from later.scripts.bridge.mini_bridge import Bridge, correct, last_boxed, load_pairs, method_hits, ngram_overlap

LORA = dict(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["q_proj", "v_proj", "o_proj"], bias="none")


def scores(P, q, Zs, execs):
    sf, ss, aux = P.exec_scores2(q, Zs, execs)
    return sf, ss, aux


def pair_loss(P, p, ZA, ZB):
    sf, ss, aux = scores(P, p["question"], torch.stack([ZA, ZB, ZB, ZA]),
                         [p["execution_A"], p["execution_B"], p["execution_A"], p["execution_B"]])
    l_ce = -(sf[0] + sf[1]) / 2
    mA, mB = ss[0] - ss[2], ss[1] - ss[3]
    l_rank = F.softplus(-mA) + F.softplus(-mB)
    return {"loss": l_ce + mb.LAMBDA_RANK * l_rank, "l_ce": l_ce, "l_rank": l_rank, "margin_A": mA, "margin_B": mB,
            "margin_A_full": sf[0] - sf[2], "margin_B_full": sf[1] - sf[3], "aux": aux}


@torch.no_grad()
def evaluate(P, pairs, Zc, Zmean, perm):
    """Per pair: s(E_X | Z) for Z in {own (correct), same-q swapped, cross-question shuffled (same label), zero, mean}."""
    rows = []
    for i, p in enumerate(pairs):
        ZA, ZB = Zc[p["id"]]
        dA, dB = Zc[pairs[perm[i]]["id"]]
        conds = [("A", "correct", ZA), ("A", "swapped", ZB), ("A", "shuffled", dA), ("A", "zero", torch.zeros_like(ZA)), ("A", "mean", Zmean),
                 ("B", "correct", ZB), ("B", "swapped", ZA), ("B", "shuffled", dB), ("B", "zero", torch.zeros_like(ZB)), ("B", "mean", Zmean)]
        sf, ss, _ = scores(P, p["question"], torch.stack([c[2] for c in conds]),
                           [p["execution_" + c[0]] for c in conds])
        r = {"id": p["id"], "shuffled_donor": pairs[perm[i]]["id"]}
        for k, (e, c, _) in enumerate(conds):
            r[f"span_s_{e}_{c}"] = float(ss[k]); r[f"full_s_{e}_{c}"] = float(sf[k])
        for e in ("A", "B"):
            for c in ("swapped", "shuffled", "zero", "mean"):
                r[f"margin_{e}_vs_{c}"] = r[f"span_s_{e}_correct"] - r[f"span_s_{e}_{c}"]
                r[f"margin_full_{e}_vs_{c}"] = r[f"full_s_{e}_correct"] - r[f"full_s_{e}_{c}"]
        rows.append(r)
    summ = {}
    for c in ("swapped", "shuffled", "zero", "mean"):
        for kind in ("", "full_"):
            v = [r[f"margin_{kind}A_vs_{c}"] for r in rows] + [r[f"margin_{kind}B_vs_{c}"] for r in rows]
            summ[f"{kind}margin_vs_{c}"] = {"mean": float(np.mean(v)), "frac_pos": float(np.mean([x > 0 for x in v])), "ci95": boot(v)}
    summ["pairs_both_pos_vs_swapped"] = sum(r["margin_A_vs_swapped"] > 0 and r["margin_B_vs_swapped"] > 0 for r in rows)
    summ["ce_full_correct"] = -float(np.mean([r["full_s_A_correct"] for r in rows] + [r["full_s_B_correct"] for r in rows]))
    summ["ce_span_correct"] = -float(np.mean([r["span_s_A_correct"] for r in rows] + [r["span_s_B_correct"] for r in rows]))
    return summ, rows


@torch.no_grad()
def generations(P, pairs, Zc, max_new):
    out = []
    for p in pairs:
        ZA, ZB = Zc[p["id"]]
        for cond, z in (("Z_A", ZA), ("Z_B", ZB)):
            g = P.generate(p["question"], Z=z, max_new_tokens=max_new)
            pred = last_boxed(g)
            out.append({"id": p["id"], "condition": cond, "generation": g, "pred": pred, "correct": correct(pred, str(p["answer"])),
                        "overlap_E_A": ngram_overlap(g, p["execution_A"]), "overlap_E_B": ngram_overlap(g, p["execution_B"]),
                        "method_terms_A_hits": method_hits(g, p["method_terms_only_A"]),
                        "method_terms_B_hits": method_hits(g, p["method_terms_only_B"]),
                        "ended": g.rstrip().endswith("<|im_end|>")})
    return out


def gen_summary(G):
    by = {}
    for r in G:
        by.setdefault(r["id"], {})[r["condition"]] = r
    return {"identical_pairs": sum(v["Z_A"]["generation"] == v["Z_B"]["generation"] for v in by.values()),
            "auto_switch_pairs": [k for k, v in by.items() if v["Z_A"]["overlap_E_A"] > v["Z_A"]["overlap_E_B"]
                                  and v["Z_B"]["overlap_E_B"] > v["Z_B"]["overlap_E_A"]],
            "correct": sum(r["correct"] for r in G), "ended": sum(r["ended"] for r in G)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="dir with train_pairs.jsonl / heldout_pairs.jsonl")
    ap.add_argument("--bridge_ckpt", required=True)
    ap.add_argument("--train_stats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pairs_per_step", type=int, default=4)
    ap.add_argument("--gen_tokens", type=int, default=1024)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0); random.seed(0)
    t_start = time.time()
    train = load_pairs(os.path.join(a.data, "train_pairs.jsonl"))
    held = load_pairs(os.path.join(a.data, "heldout_pairs.jsonl"))
    P = PilotABCD(a.model, "std", 64)
    d = P.model.config.hidden_size
    sanity = {}
    # ---- frozen B-50 bridge + saved train-only stats (verify they equal a fresh train-only fit)
    st = np.load(a.train_stats)
    fit = P.fit_stats(train)
    sanity["train_stats_reproduce_saved"] = bool(np.allclose(fit["mu"].cpu().numpy(), st["mu"], atol=1e-6, rtol=1e-5) and
                                                 np.allclose(fit["sigma"].cpu().numpy(), st["sigma"], atol=1e-6, rtol=1e-5) and
                                                 float(st["floor"]) == fit["floor"])
    P.stats = {"mu": torch.tensor(st["mu"], device=P.dev), "sigma": torch.tensor(st["sigma"], device=P.dev),
               "floor": float(st["floor"]), "n_tokens": int(st["n_tokens"]), "frac_dims_floored": fit["frac_dims_floored"]}
    bridge = Bridge(d, P.tok_rms).to(P.dev)
    ck = torch.load(a.bridge_ckpt)
    assert ck["input_mode"] == "std" and ck["step"] == 50
    bridge.load_state_dict(ck["bridge"]); bridge.eval()
    for q in bridge.parameters():
        q.requires_grad_(False)
    bridge_snap = {k: v.clone() for k, v in bridge.state_dict().items()}
    stats_snap = {k: v.clone() for k, v in P.stats.items() if torch.is_tensor(v)}
    # ---- cache Z with the frozen, LoRA-free model
    with torch.no_grad():
        Zc = {}
        for p in train + held:
            Z = P.Z(bridge, p["question"], [p["plan_A"], p["plan_B"]])
            Zc[p["id"]] = (Z[0].clone(), Z[1].clone())
        Zmean = torch.stack([z for p in train for z in Zc[p["id"]]]).mean(0)
        Zmean = Zmean * (P.tok_rms / Zmean.pow(2).mean(-1, keepdim=True).sqrt())   # same RMS as learned latents
    torch.save({k: torch.stack(v).cpu() for k, v in Zc.items()}, os.path.join(a.out, "cached_Z.pt"))
    perm_h = [(i + 1) % len(held) for i in range(len(held))]      # fixed cross-question donor (derangement)
    perm_t = [(i + 1) % len(train) for i in range(len(train))]
    # ---- step-0 baseline (no LoRA attached)
    h0, h0rows = evaluate(P, held, Zc, Zmean, perm_h)
    t0s, _ = evaluate(P, train, Zc, Zmean, perm_t)
    g0 = generations(P, held, Zc, a.gen_tokens)
    json.dump({"step": 0, "lora": "not attached", "heldout": h0, "heldout_rows": h0rows, "train": t0s, "gen_summary": gen_summary(g0),
               "z_cos_AB_heldout": float(np.mean([F.cosine_similarity(*Zc[p["id"]], dim=-1).mean().item() for p in held]))},
              open(os.path.join(a.out, "heldout_step0.json"), "w"), indent=1)
    print("STEP0 held span margin vs swapped %.5f (frac %.2f) | gens %s" % (h0["margin_vs_swapped"]["mean"], h0["margin_vs_swapped"]["frac_pos"], gen_summary(g0)), flush=True)
    with torch.no_grad():   # reference for the LoRA-init identity check: SAME single-sequence batch as after attaching
        p = held[0]
        ref_single = float(scores(P, p["question"], torch.stack([Zc[p["id"]][0]]), [p["execution_A"]])[0][0])
    # ---- attach LoRA (in place on the Qwen3 modules); P.model stays the same object
    base_snap = {k: v.detach().cpu().clone() for k, v in P.model.state_dict().items()}
    peft_model = get_peft_model(P.model, LoraConfig(**LORA))
    lora_params = [(n, q) for n, q in peft_model.named_parameters() if q.requires_grad]
    total = sum(q.numel() for q in peft_model.parameters())
    sanity["trainable_params"] = sum(q.numel() for _, q in lora_params)
    sanity["total_params_executor"] = total
    sanity["trainable_all_lora"] = all("lora_" in n for n, _ in lora_params)
    sanity["trainable_module_examples"] = sorted({n.split(".lora_")[0].split(".")[-1] for n, _ in lora_params})
    sanity["bridge_trainable_params"] = sum(q.numel() for q in bridge.parameters() if q.requires_grad)
    # LoRA B=0 at init -> identical scores; and online Z with adapter disabled == cache
    with torch.no_grad():
        p = held[0]
        sf_l, _, _ = scores(P, p["question"], torch.stack([Zc[p["id"]][0]]), [p["execution_A"]])
        sanity["lora_init_score_abs_diff"] = abs(float(sf_l[0]) - ref_single)
        sanity["lora_init_score_equals_step0"] = sanity["lora_init_score_abs_diff"] < 1e-6
        with peft_model.disable_adapter():
            Zon = P.Z(bridge, p["question"], [p["plan_A"], p["plan_B"]])
        sanity["online_Z_adapter_disabled_equals_cache_max_abs"] = float(torch.stack([(Zon[0] - Zc[p["id"]][0]).abs().max(),
                                                                                      (Zon[1] - Zc[p["id"]][1]).abs().max()]).max())
    opt = torch.optim.AdamW([q for _, q in lora_params], lr=a.lr, weight_decay=0.01)
    sanity["optimizer_param_count"] = sum(q.numel() for g in opt.param_groups for q in g["params"])
    # ---- smoke: 1 batch x 2 steps (on a copy of the optimizer state? no: smoke uses a throwaway optimizer & restores LoRA)
    lora_init = {n: q.detach().clone() for n, q in lora_params}
    sopt = torch.optim.AdamW([q for _, q in lora_params], lr=a.lr, weight_decay=0.01)
    smoke = []
    for s in range(2):
        sopt.zero_grad(set_to_none=True)
        t = pair_loss(P, train[0], *Zc[train[0]["id"]])
        t["loss"].backward()
        gs = [q.grad for _, q in lora_params]
        smoke.append({"step": s, "loss": float(t["loss"]), "grad_finite": all(g is not None and torch.isfinite(g).all().item() for g in gs),
                      "grad_norm": float(torch.sqrt(sum(g.pow(2).sum() for g in gs if g is not None))),
                      "slots_equal_Z": float((t["aux"]["emb"][:, t["aux"]["slots"]].float() - torch.stack(
                          [Zc[train[0]["id"]][i] for i in (0, 1, 1, 0)]).to(torch.bfloat16).float()).abs().max()) == 0.0,
                      "nonlora_grads_none": all(q.grad is None for n, q in peft_model.named_parameters() if "lora_" not in n)})
        sopt.step()
    sanity["smoke"] = smoke
    sanity["lora_changed_by_smoke"] = any(not torch.equal(q, lora_init[n]) for n, q in lora_params)
    with torch.no_grad():
        for n, q in lora_params:
            q.copy_(lora_init[n])                       # restore exact init; real run starts from the same init
    sd = {k.replace("base_model.model.", "").replace(".base_layer", ""): v for k, v in peft_model.state_dict().items() if "lora_" not in k}
    sanity["backbone_bitwise_unchanged_after_smoke"] = all(torch.equal(sd[k].cpu(), base_snap[k]) for k in base_snap)
    sanity["bridge_bitwise_unchanged"] = all(torch.equal(v, bridge_snap[k]) for k, v in bridge.state_dict().items())
    sanity["stats_unchanged"] = all(torch.equal(P.stats[k], v) for k, v in stats_snap.items())
    # train vs generate path (LoRA active): first token argmax equal
    with torch.no_grad():
        p = held[0]
        _, _, aux = scores(P, p["question"], torch.stack([Zc[p["id"]][0]]), [p["execution_A"]])
        pre, slots = P.prefix_ids(p["question"])
        ids = torch.tensor([pre], device=P.dev)
        emb = P.model.get_input_embeddings()(ids).clone(); emb[0, slots] = Zc[p["id"]][0].to(emb.dtype)
        g = P.model.generate(inputs_embeds=emb, attention_mask=torch.ones_like(ids), max_new_tokens=1, do_sample=False,
                             output_scores=True, return_dict_in_generate=True, pad_token_id=P.T.pad_id)
        sanity["train_vs_generate_first_token_equal"] = int(aux["first_logits"].argmax()) == int(g.scores[0][0].argmax())
    sanity["executor_prefix_has_no_plan_text"] = P.tok.decode(P.prefix_ids(held[0]["question"])[0]).find(held[0]["plan_A"][40:120]) < 0
    hard = [sanity["train_stats_reproduce_saved"], sanity["trainable_all_lora"], sanity["bridge_trainable_params"] == 0,
            sanity["lora_init_score_equals_step0"], sanity["online_Z_adapter_disabled_equals_cache_max_abs"] == 0.0,
            sanity["optimizer_param_count"] == sanity["trainable_params"],
            all(s["grad_finite"] and s["grad_norm"] > 0 and s["slots_equal_Z"] and s["nonlora_grads_none"] for s in smoke),
            sanity["lora_changed_by_smoke"], sanity["backbone_bitwise_unchanged_after_smoke"], sanity["bridge_bitwise_unchanged"],
            sanity["stats_unchanged"], sanity["train_vs_generate_first_token_equal"], sanity["executor_prefix_has_no_plan_text"]]
    sanity["ALL_HARD_GATES_PASS"] = all(hard)
    json.dump(sanity, open(os.path.join(a.out, "sanity_checks.json"), "w"), indent=1)
    print("SANITY", json.dumps({k: v for k, v in sanity.items() if k != "smoke"}), flush=True)
    if not sanity["ALL_HARD_GATES_PASS"]:
        raise SystemExit("sanity gates failed")
    # ---- training: 50 steps, same order / relabel RNG as the bridge runs
    log = open(os.path.join(a.out, "train_log.jsonl"), "w")
    rng = random.Random(1234)
    order, step, t0 = [], 0, time.time()
    torch.cuda.reset_peak_memory_stats()
    peft_model.train()
    def ev(s, with_gen):
        peft_model.eval()
        hs, hrows = evaluate(P, held, Zc, Zmean, perm_h)
        ts, _ = evaluate(P, train, Zc, Zmean, perm_t)
        rec = {"step": s, "heldout": hs, "heldout_rows": hrows, "train": ts}
        if with_gen:
            G = generations(P, held, Zc, a.gen_tokens)
            rec["gen_summary"] = gen_summary(G)
            with open(os.path.join(a.out, f"generations_step{s}.jsonl"), "w") as f:
                for r in G:
                    f.write(json.dumps(r) + "\n")
        rec["z_cos_AB_heldout"] = float(np.mean([F.cosine_similarity(*Zc[p["id"]], dim=-1).mean().item() for p in held]))
        json.dump(rec, open(os.path.join(a.out, f"heldout_step{s}.json"), "w"), indent=1)
        log.write(json.dumps({"step": s, "eval": True, "heldout": hs, "train": ts}) + "\n"); log.flush()
        print("EVAL %d held vs_swapped %.5f (%.2f) vs_shuf %.5f vs_mean %.5f | train vs_swapped %.5f (%.2f) | ce %.4f" % (
            s, hs["margin_vs_swapped"]["mean"], hs["margin_vs_swapped"]["frac_pos"], hs["margin_vs_shuffled"]["mean"],
            hs["margin_vs_mean"]["mean"], ts["margin_vs_swapped"]["mean"], ts["margin_vs_swapped"]["frac_pos"], hs["ce_full_correct"]), flush=True)
        peft_model.train()
    while step < a.steps:
        if not order:
            idx = list(range(len(train))); rng.shuffle(idx); order = idx
        batch = [order.pop() for _ in range(min(a.pairs_per_step, len(order)))]
        opt.zero_grad(set_to_none=True)
        agg = {k: 0.0 for k in ("loss", "l_ce", "l_rank", "margin_A", "margin_B", "margin_A_full", "margin_B_full")}
        for i in batch:
            p = train[i]
            ZA, ZB = Zc[p["id"]]
            if rng.random() < 0.5:
                p = {**p, "plan_A": p["plan_B"], "execution_A": p["execution_B"], "plan_B": p["plan_A"], "execution_B": p["execution_A"]}
                ZA, ZB = ZB, ZA
            t = pair_loss(P, p, ZA, ZB)
            (t["loss"] / len(batch)).backward()
            for k in agg:
                agg[k] += float(t[k]) / len(batch)
        gn = float(torch.nn.utils.clip_grad_norm_([q for _, q in lora_params], 1.0))
        opt.step()
        step += 1
        log.write(json.dumps({"step": step, **agg, "grad_norm": gn, "pairs": [train[i]["id"] for i in batch],
                              "elapsed_s": time.time() - t0, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}) + "\n"); log.flush()
        if step % 5 == 0:
            print("step", step, {k: round(v, 5) for k, v in agg.items()}, "gn %.3f" % gn, flush=True)
        if step == 25:
            ev(25, with_gen=False)
    train_time, peak = time.time() - t0, torch.cuda.max_memory_allocated() / 2**30
    peft_model.save_pretrained(os.path.join(a.out, "lora_adapter_step50"))
    ev(50, with_gen=True)
    sd = {k.replace("base_model.model.", "").replace(".base_layer", ""): v for k, v in peft_model.state_dict().items() if "lora_" not in k}
    post = {"backbone_bitwise_unchanged_after_training": all(torch.equal(sd[k].cpu(), base_snap[k]) for k in base_snap),
            "bridge_bitwise_unchanged_after_training": all(torch.equal(v, bridge_snap[k]) for k, v in bridge.state_dict().items()),
            "train_time_s": train_time, "peak_mem_gb": peak, "total_wall_s": time.time() - t_start}
    json.dump(post, open(os.path.join(a.out, "post_training_checks.json"), "w"), indent=1)
    print("POST", json.dumps(post), flush=True)


if __name__ == "__main__":
    main()
