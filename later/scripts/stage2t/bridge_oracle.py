"""Stage 2B oracle teacher-latent -> executor bridge smoke test (5-10 steps).

Init = joint 2-loss step-50 (LoRA trainable, projector + base frozen); input = Q + 8 frozen 2A teacher latents Z_T(P_X)
in the existing slots; loss = 0.5 * [CE(E_A | Q, Z_A) + CE(E_B | Q, Z_B)] over E + '</EXECUTION><|im_end|>' (the
existing 2B target), all verified train pairs every step.  Eval (train / held-out): matched CE, Delta_match on the full
target, on the strategy span (after the E_A/E_B token-LCP, before the last '\\boxed') and on the key prefix (first 24
tokens after the LCP), plus batched greedy generation under Z_A and Z_B.
Modes: audit | train (--stop_at, --resume) | eval (--ckpt).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from later.scripts.bridge.bridge_input_abcd import boot
from later.scripts.bridge.mini_bridge import correct, last_boxed, method_hits, ngram_overlap
from later.scripts.stage2s.stochastic_pilot import grad_norm, short
from later.scripts.stage2t.contrastive_2b import exec_forward, generate_batch, norm_q, train_pairs
from later.scripts.stage2t.teacher_vs_student_0step import strategy_span
from later.scripts.stage2t.three_step import TStudent, load_jsonl, save_ck


def scores(S, p):
    """2x2 on full target / strategy span / key prefix.  rows: (Z_A,E_A) (Z_B,E_A) (Z_A,E_B) (Z_B,E_B)."""
    TA, TB = S.teacher(p["plan_A_norm"]), S.teacher(p["plan_B_norm"])
    spA, _ = strategy_span(S, p, "A"); spB, _ = strategy_span(S, p, "B"); k = len(p["C"])
    kpA, kpB = (k, k + len(p["D_A"])), (k, k + len(p["D_B"]))
    Z = torch.stack([TA, TB, TA, TB]); tg = [p["tgt_A"], p["tgt_A"], p["tgt_B"], p["tgt_B"]]
    sf, ss, _, _ = exec_forward(S, p["question"], Z, tg, [spA, spA, spB, spB])
    _, sk, _, _ = exec_forward(S, p["question"], Z, tg, [kpA, kpA, kpB, kpB])
    out = {}
    for nm, s in (("full", sf), ("span", ss), ("key", sk)):
        s = [float(x) for x in s]
        out[nm] = {"AA": s[0], "BA": s[1], "AB": s[2], "BB": s[3], "delta_match": ((s[0] - s[1]) + (s[3] - s[2])) / 2,
                   "delta_A": s[0] - s[1], "delta_B": s[3] - s[2]}
    out["matched_ce"] = -(out["full"]["AA"] + out["full"]["BB"]) / 2
    return out


def bridge_loss(S, p):
    TA, TB = S.teacher(p["plan_A_norm"]), S.teacher(p["plan_B_norm"])
    sf, _, _, _ = exec_forward(S, p["question"], torch.stack([TA, TB]), [p["tgt_A"], p["tgt_B"]])
    return -0.5 * (sf[0] + sf[1])


def label(g, p):
    oa, ob = ngram_overlap(g, p["execution_A"]), ngram_overlap(g, p["execution_B"])
    return "A" if oa > ob else ("B" if ob > oa else "=")


def evaluate(S, tr, ho, out, tag, gen_tokens, bs):
    res = {"tag": tag}; rows = []
    with torch.no_grad():
        for role, P in (("train", tr), ("heldout", ho)):
            for p in P:
                rows.append({"id": short(p["id"]), "role": role, "label_A": p.get("method_label_A"), "label_B": p.get("method_label_B"), **scores(S, p)})
        allp = tr + ho
        items = [(p, X) for p in allp for X in ("A", "B")]
        gens = []
        for i in range(0, len(items), bs):
            ch = items[i:i + bs]
            Z = torch.stack([S.teacher(p["plan_" + X + "_norm"]) for p, X in ch])
            gens += generate_batch(S, [p["question"] for p, _ in ch], Z, gen_tokens)
    G = {}
    for (p, X), g in zip(items, gens):
        pr = last_boxed(g)
        G[(short(p["id"]), X)] = {"label": label(g, p), "ov_A": ngram_overlap(g, p["execution_A"]), "ov_B": ngram_overlap(g, p["execution_B"]),
                                  "terms_A": method_hits(g, p["method_terms_only_A"]), "terms_B": method_hits(g, p["method_terms_only_B"]),
                                  "pred": pr, "correct": correct(pr, str(p["answer"])), "format_ok": "</EXECUTION>" in g and "<|im_end|>" in g, "text": g}
    for r in rows:
        r["gen_ZA"], r["gen_ZB"] = G[(r["id"], "A")], G[(r["id"], "B")]
        r["auto_switch"] = r["gen_ZA"]["label"] == "A" and r["gen_ZB"]["label"] == "B"
        r["identical_AB"] = r["gen_ZA"]["text"] == r["gen_ZB"]["text"]
    for role in ("train", "heldout"):
        R = [r for r in rows if r["role"] == role]
        o = {"n_pairs": len(R), "matched_ce": float(np.mean([r["matched_ce"] for r in R]))}
        for nm in ("full", "span", "key"):
            v = [r[nm]["delta_match"] for r in R]
            o[f"delta_match_{nm}"] = {"mean": float(np.mean(v)), "ci95": boot(v), "n_pos": int(sum(x > 0 for x in v))}
            o[f"both_dirs_pos_{nm}"] = int(sum(r[nm]["delta_A"] > 0 and r[nm]["delta_B"] > 0 for r in R))
        o["auto_switch"] = int(sum(r["auto_switch"] for r in R)); o["identical_AB_text"] = int(sum(r["identical_AB"] for r in R))
        o["auto_labels_ZA"] = "".join(r["gen_ZA"]["label"] for r in R); o["auto_labels_ZB"] = "".join(r["gen_ZB"]["label"] for r in R)
        o["correct"] = f"{sum(r['gen_ZA']['correct'] for r in R) + sum(r['gen_ZB']['correct'] for r in R)}/{2 * len(R)}"
        o["format_ok"] = f"{sum(r['gen_ZA']['format_ok'] for r in R) + sum(r['gen_ZB']['format_ok'] for r in R)}/{2 * len(R)}"
        res[role] = o
    os.makedirs(f"{out}/evaluations", exist_ok=True)
    json.dump(res, open(f"{out}/evaluations/eval_{tag}.json", "w"), indent=1)
    with open(f"{out}/evaluations/rows_{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("EVAL", json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["audit", "train", "eval"], required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--s2b", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--init", required=True); ap.add_argument("--resume", default=None); ap.add_argument("--ckpt", default=None)
    ap.add_argument("--stop_at", type=int, default=5); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gen_tokens", type=int, default=2048); ap.add_argument("--gen_bs", type=int, default=27)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(0)
    S = TStudent(a.model, a.s2b, seed=0)
    ck0 = torch.load(a.init, weights_only=False); S.load(ck0["state"])
    for _, p in S.proj:
        p.requires_grad_(False)
    lora = [p for _, p in S.lora]
    tr, ho, allp = train_pairs(S, a.s2b)
    print("INIT", a.init, "step", ck0.get("step"), "phase", ck0.get("phase"), "| train pairs", len(tr), "held-out", len(ho), flush=True)

    if a.mode == "audit":
        g = {"init": a.init, "init_step": ck0.get("step"), "lr": a.lr}
        g["requires_grad"] = {"lora": sum(p.numel() for p in lora if p.requires_grad), "projector": sum(p.numel() for _, p in S.proj if p.requires_grad),
                              "base": sum(p.numel() for n, p in S.peft.named_parameters() if p.requires_grad and "lora_" not in n),
                              "encoder": "frozen 2A step-300 compressor; Z_T read from the verified z_cache (online-vs-cache max |diff| 2.4e-4)"}
        g["data"] = {"train_pairs": [short(p["id"]) for p in tr], "heldout_pairs": [short(p["id"]) for p in ho],
                     "excluded": [short(p["id"]) + ": " + p["excluded"] for p in allp if p["excluded"]],
                     "question_overlap_train_heldout": len({norm_q(p["question"]) for p in tr} & {norm_q(p["question"]) for p in ho})}
        zs = []
        for p in tr[:2]:
            for X in ("A", "B"):
                Z = S.teacher(p["plan_" + X + "_norm"])
                zs.append({"id": short(p["id"]), "plan": X, "shape": list(Z.shape), "finite": bool(torch.isfinite(Z).all()), "nonzero": bool((Z.abs() > 0).any()),
                           "slot_rms": [round(float(x), 6) for x in Z.pow(2).mean(-1).sqrt()]})
        g["teacher_latents_sample"] = zs; g["tok_rms"] = S.tok_rms
        # injection + prefix: no plan / target text in the prefix
        p = tr[0]; TA, TB = S.teacher(p["plan_A_norm"]), S.teacher(p["plan_B_norm"])
        with torch.no_grad():
            _, _, _, ax = exec_forward(S, p["question"], torch.stack([TA, TB]), [p["tgt_A"], p["tgt_B"]])
        sl = ax["slots"]; pre = S.tok.decode(ax["ids"][0, :ax["n_pre"]])
        g["injection"] = {"slots_hold_ZA_ZB": bool(torch.equal(ax["emb"][0, sl], TA.to(ax["emb"].dtype)) and torch.equal(ax["emb"][1, sl], TB.to(ax["emb"].dtype))),
                          "prefix_ids_identical": bool(torch.equal(ax["ids"][0, :ax["n_pre"]], ax["ids"][1, :ax["n_pre"]])),
                          "no_plan_or_exec_in_prefix": all(t[10:80] not in pre for t in (p["plan_A_norm"], p["plan_B_norm"], p["execution_A"], p["execution_B"])),
                          "no_method_label_in_prefix": not any(lb and lb in pre for lb in (p.get("method_label_A"), p.get("method_label_B"))),
                          "prefix_tail": S.tok.decode(ax["ids"][0, ax["n_pre"] - 12:ax["n_pre"]]), "target_tail": S.tok.decode(p["tgt_A"][-5:])}
        # gradient routing
        for q_ in S.peft.parameters():
            q_.grad = None
        bridge_loss(S, p).backward()
        g["grads"] = {"lora": grad_norm(lora), "projector": grad_norm([q_ for _, q_ in S.proj]),
                      "base_any_grad": any(q_.grad is not None for n, q_ in S.peft.named_parameters() if "lora_" not in n and "latent_projector" not in n),
                      "finite": all(torch.isfinite(q_.grad).all().item() for q_ in lora if q_.grad is not None)}
        g["PASS"] = g["requires_grad"]["projector"] == 0 and g["requires_grad"]["base"] == 0 and g["data"]["question_overlap_train_heldout"] == 0 and \
            all(z["finite"] and z["nonzero"] and z["shape"] == [8, 2560] for z in zs) and all(v for k, v in g["injection"].items() if not k.endswith("tail")) and \
            g["grads"]["lora"] > 0 and g["grads"]["projector"] == 0 and not g["grads"]["base_any_grad"] and g["grads"]["finite"]
        json.dump(g, open(f"{a.out}/audit.json", "w"), indent=1); print(json.dumps(g, indent=1))
        return

    if a.mode == "eval":
        if a.ckpt:
            S.load(torch.load(a.ckpt, weights_only=False)["state"])
        evaluate(S, tr, ho, a.out, os.path.basename(a.ckpt or a.init).replace(".pt", "") if a.ckpt else "step0", a.gen_tokens, a.gen_bs)
        return

    opt = torch.optim.AdamW(lora, lr=a.lr, weight_decay=0.0)
    done = 0
    if a.resume:
        ck = torch.load(a.resume, weights_only=False); S.load(ck["state"]); opt.load_state_dict(ck["opt"]); done = ck["step"]
    os.makedirs(f"{a.out}/checkpoints", exist_ok=True); log = open(f"{a.out}/train.jsonl", "a"); t0 = time.time()
    for step in range(done + 1, a.stop_at + 1):
        opt.zero_grad(set_to_none=True); ls = []
        for p in tr:                                   # every verified train pair, A and B balanced
            l = bridge_loss(S, p); (l / len(tr)).backward(); ls.append(float(l))
        gl, gp = grad_norm(lora), grad_norm([q_ for _, q_ in S.proj])
        fin = all(torch.isfinite(q_.grad).all().item() for q_ in lora if q_.grad is not None)
        gn = float(torch.nn.utils.clip_grad_norm_(lora, 1.0)); opt.step()
        r = {"step": step, "loss_bridge": float(np.mean(ls)), "grad_lora": gl, "grad_projector": gp, "grad_preclip": gn, "finite": fin,
             "elapsed_s": time.time() - t0, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
        log.write(json.dumps(r) + "\n"); log.flush(); print("STEP", json.dumps(r), flush=True)
        if not fin:
            print("STOP non-finite"); break
    save_ck(S, opt, f"{a.out}/checkpoints/bridge_step{step}.pt", step, {"init": a.init, "lr": a.lr, "loss": "0.5*(CE(E_A|Q,Z_A)+CE(E_B|Q,Z_B))"})


if __name__ == "__main__":
    main()
