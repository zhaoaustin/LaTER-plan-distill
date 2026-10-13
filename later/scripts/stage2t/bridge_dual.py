"""Dual-initialization Teacher-Latent Bridge SFT pilot (one shared data set, only the init differs).

Run A: Stage-1 explicit SFT (full-param, frozen) + a FRESH LoRA r16 q/v/o (seed 0, lora_B = 0 -> exactly Stage-1 at step 0).
Run B: Stage-1 + the Stage-2B joint step-50 LoRA (no bridge CE before).  Projector frozen and not in the forward in both.
Input: chat(Q) + '<EXPLORATION>\\n' + [Z_T(P_X) in the 8 slots] + '</EXPLORATION><EXECUTION>\\n'; Z_T = frozen 2A step-300
compressor output read from ONE shared latent file.  Loss = 1.0 * CE(E_X + '</EXECUTION><|im_end|>' | Q, Z_T(P_X)), per-token
mean per sequence, (A + B) / 2 per question, every train question every step (the verified bridge-CE config), LoRA only.
Modes: audit | train (checkpoints at --save_steps) | eval (--ckpt or step 0): 2x2 scores on train / held-out, held-out and
train generation under Z_A / Z_B, paraphrase latents (train questions), normal explicit mode on fixed text questions.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from later.scripts.bridge.bridge_input_abcd import boot
from later.scripts.bridge.mini_bridge import correct, last_boxed, method_hits, ngram_overlap
from later.scripts.stage2s.stochastic_pilot import SPAN, grad_norm, short
from later.scripts.stage2t.bridge_oracle import scores
from later.scripts.stage2t.contrastive_2b import exec_forward, generate_batch, norm_q
from later.scripts.stage2t.three_step import TStudent, load_jsonl, save_ck
from later.src.train.stage2_plan_dataset import render_prompt

key = lambda s: hashlib.sha1(s.encode()).hexdigest()


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()[:16]


def prep(S, recs):
    out = []
    for p in recs:
        a = S.tok.encode(p["execution_A"], add_special_tokens=False); b = S.tok.encode(p["execution_B"], add_special_tokens=False)
        k = 0
        while k < min(len(a), len(b)) and a[k] == b[k]:
            k += 1
        q = dict(p)
        q.update({"C": a[:k], "D_A": a[k:k + SPAN], "D_B": b[k:k + SPAN], "tgt_A": S.target_ids(p["execution_A"]), "tgt_B": S.target_ids(p["execution_B"])})
        assert q["tgt_A"][:k + len(q["D_A"])] == q["C"] + q["D_A"] and q["tgt_B"][:k + len(q["D_B"])] == q["C"] + q["D_B"], p["id"]
        out.append(q)
    return out


def bridge_loss(S, p):
    TA, TB = S.teacher(p["plan_A_norm"]), S.teacher(p["plan_B_norm"])
    sf, _, _, _ = exec_forward(S, p["question"], torch.stack([TA, TB]), [p["tgt_A"], p["tgt_B"]])
    return -0.5 * (sf[0] + sf[1])


def label(g, ea, eb):
    oa, ob = ngram_overlap(g, ea), ngram_overlap(g, eb)
    return "A" if oa > ob else ("B" if ob > oa else "=")


def gen_rows(S, items, gen_tokens, bs):
    """items: (p, latent_name, Z, target_strategy) -> generation rows (auto label vs gold E_A / E_B)."""
    gens = []
    for i in range(0, len(items), bs):
        ch = items[i:i + bs]
        gens += generate_batch(S, [p["question"] for p, _, _, _ in ch], torch.stack([z for _, _, z, _ in ch]), gen_tokens)
    rows = []
    for (p, nm, _, tgt), g in zip(items, gens):
        pr = last_boxed(g)
        rows.append({"id": short(p["id"]), "latent": nm, "target": tgt, "auto_label": label(g, p["execution_A"], p["execution_B"]),
                     "ov_A": ngram_overlap(g, p["execution_A"]), "ov_B": ngram_overlap(g, p["execution_B"]),
                     "terms_A": method_hits(g, p.get("method_terms_only_A") or []), "terms_B": method_hits(g, p.get("method_terms_only_B") or []),
                     "pred": pr, "correct": correct(pr, str(p["answer"])), "format_ok": "</EXECUTION>" in g and "<|im_end|>" in g,
                     "n_tokens": len(S.tok.encode(g, add_special_tokens=False)), "text": g})
    return rows


@torch.no_grad()
def normal_mode(S, qs, max_new, bs):
    """Stage-1 explicit mode: chat(Q) -> model writes <EXPLORATION> plan + <EXECUTION>; greedy, left-padded batches, fresh KV."""
    out = []
    for i in range(0, len(qs), bs):
        ch = qs[i:i + bs]
        pres = [S.tok.encode(render_prompt(S.tok, r["question"]), add_special_tokens=False) for r in ch]
        n = max(map(len, pres)); K = len(ch)
        ids = torch.full((K, n), S.T.pad_id, dtype=torch.long); att = torch.zeros(K, n, dtype=torch.long)
        for j, p in enumerate(pres):
            ids[j, n - len(p):] = torch.tensor(p); att[j, n - len(p):] = 1
        ids, att = ids.to(S.dev), att.to(S.dev); pos = (att.cumsum(1) - 1).clamp_min(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = S.base.model(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=True)
            cache = o.past_key_values; lg = S.base.lm_head(o.last_hidden_state[:, -1]).float()
            eos = torch.tensor(S.eos, device=S.dev); done = torch.zeros(K, dtype=torch.bool, device=S.dev); gen = [[] for _ in range(K)]; last = pos[:, -1:]
            for _ in range(max_new):
                tk = lg.argmax(-1)
                for j in range(K):
                    if not done[j]:
                        gen[j].append(int(tk[j]))
                done = done | torch.isin(tk, eos)
                if bool(done.all()):
                    break
                att = torch.cat([att, torch.ones(K, 1, dtype=att.dtype, device=S.dev)], 1); last = last + 1
                o = S.base.model(input_ids=tk.unsqueeze(1), attention_mask=att, position_ids=last, past_key_values=cache, use_cache=True)
                cache = o.past_key_values; lg = S.base.lm_head(o.last_hidden_state[:, -1]).float()
        for r, g in zip(ch, gen):
            t = S.tok.decode(g, skip_special_tokens=False); pr = last_boxed(t)
            out.append({"id": short(r["record_id"]), "pred": pr, "correct": correct(pr, str(r["answer"])),
                        "format_ok": t.startswith("<EXPLORATION>") and "</EXPLORATION><EXECUTION>" in t and "</EXECUTION>" in t and "<|im_end|>" in t,
                        "n_tokens": len(g), "text": t})
    return out


def summarize_scores(R):
    o = {"n": len(R), "matched_ce": float(np.mean([r["matched_ce"] for r in R]))}
    for nm in ("full", "span", "key"):
        dA = [r[nm]["delta_A"] for r in R]; dB = [r[nm]["delta_B"] for r in R]
        o[nm] = {"mean_delta_A": float(np.mean(dA)), "mean_delta_B": float(np.mean(dB)), "ci95_delta_A": boot(dA), "ci95_delta_B": boot(dB),
                 "both_pos": int(sum(a > 0 and b > 0 for a, b in zip(dA, dB))), "frac_both_pos": float(np.mean([a > 0 and b > 0 for a, b in zip(dA, dB)]))}
    return o


def summarize_gen(rows, pa="Z_A", pb="Z_B"):
    by = {}
    for r in rows:
        by.setdefault(r["id"], {})[r["latent"]] = r
    ids = [i for i in by if pa in by[i] and pb in by[i]]
    A = [by[i][pa] for i in ids]; B = [by[i][pb] for i in ids]
    return {"n_q": len(ids), "pair_follow_auto": int(sum(a["auto_label"] == "A" and b["auto_label"] == "B" for a, b in zip(A, B))),
            "A_adherence_auto": int(sum(a["auto_label"] == "A" for a in A)), "B_adherence_auto": int(sum(b["auto_label"] == "B" for b in B)),
            "identical_text": int(sum(a["text"] == b["text"] for a, b in zip(A, B))),
            "correct_A": int(sum(a["correct"] for a in A)), "correct_B": int(sum(b["correct"] for b in B)),
            "format_A": int(sum(a["format_ok"] for a in A)), "format_B": int(sum(b["format_ok"] for b in B)),
            "labels_A": "".join(a["auto_label"] for a in A), "labels_B": "".join(b["auto_label"] for b in B)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["audit", "train", "eval"], required=True)
    ap.add_argument("--run", choices=["A", "B"], required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--s2b", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--init", default=None, help="run B only: Stage-2B joint step-50 checkpoint")
    ap.add_argument("--out", required=True); ap.add_argument("--ckpt", default=None)
    ap.add_argument("--steps", type=int, default=150); ap.add_argument("--save_steps", default="25,50,100,150"); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gen_tokens", type=int, default=2048); ap.add_argument("--gen_bs", type=int, default=48)
    ap.add_argument("--normal_tokens", type=int, default=3072); ap.add_argument("--train_gen", type=int, default=1)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(0)
    assert (a.run == "A") == (a.init is None), "A = fresh LoRA on Stage 1 (no init); B = joint step-50 init"
    S = TStudent(a.model, a.s2b, seed=0)                       # fresh LoRA (lora_B = 0) + fresh projector, seed 0
    lora0_fresh = {n: p.detach().clone() for n, p in S.lora}
    ck0 = None
    if a.init:
        ck0 = torch.load(a.init, weights_only=False); S.load(ck0["state"])
    for _, p in S.proj:
        p.requires_grad_(False)
    lora = [p for _, p in S.lora]
    LZ = torch.load(f"{a.data}/stage2a_teacher_latents.pt", weights_only=False)
    S.Zraw = LZ["Z"]                                           # the ONE shared latent cache (both runs)
    tr = prep(S, load_jsonl(f"{a.data}/bridge_dual_train.jsonl")); ho = prep(S, load_jsonl(f"{a.data}/bridge_dual_heldout.jsonl"))
    para = load_jsonl(f"{a.data}/bridge_dual_paraphrase_eval.jsonl"); normal = load_jsonl(f"{a.data}/normal_mode_questions.jsonl")
    print("RUN", a.run, "init", a.init or "Stage-1 + fresh LoRA", "| train", len(tr), "held-out", len(ho), "para", len(para), flush=True)

    if a.mode == "audit":
        g = {"run": a.run, "model": a.model, "init": a.init, "init_sha16": sha16(a.init) if a.init else None, "init_step": ck0.get("step") if ck0 else None,
             "latent_file": f"{a.data}/stage2a_teacher_latents.pt", "latent_file_sha16": sha16(f"{a.data}/stage2a_teacher_latents.pt"),
             "data_sha16": {f: sha16(f"{a.data}/{f}") for f in ("bridge_dual_train.jsonl", "bridge_dual_heldout.jsonl", "bridge_dual_paraphrase_eval.jsonl", "normal_mode_questions.jsonl")}}
        g["lora"] = {"n_tensors": len(lora), "numel": sum(p.numel() for p in lora),
                     "lora_B_abs_sum": float(sum(p.abs().sum() for n, p in S.lora if "lora_B" in n)),
                     "max_abs_diff_vs_fresh": max(float((p - lora0_fresh[n]).abs().max()) for n, p in S.lora),
                     "equals_init_ckpt": all(torch.equal(p, ck0["state"]["lora"][n].to(p.device)) for n, p in S.lora) if ck0 else None}
        g["requires_grad"] = {"lora": sum(p.numel() for p in lora if p.requires_grad), "projector": sum(p.numel() for _, p in S.proj if p.requires_grad),
                              "base": sum(p.numel() for n, p in S.peft.named_parameters() if p.requires_grad and "lora_" not in n)}
        g["data"] = {"train": len(tr), "heldout": len(ho), "question_overlap": len({norm_q(p["question"]) for p in tr} & {norm_q(p["question"]) for p in ho}),
                     "train_ids_order": [short(p["id"]) for p in tr], "heldout_ids": [short(p["id"]) for p in ho]}
        zs = []
        for p in tr + ho:
            TA, TB = S.teacher(p["plan_A_norm"]), S.teacher(p["plan_B_norm"])
            zs.append({"id": short(p["id"]), "cos_AB": float(torch.nn.functional.cosine_similarity(TA.flatten(), TB.flatten(), 0)), "max_abs_AB": float((TA - TB).abs().max()),
                       "rms_A": float(TA.pow(2).mean().sqrt()), "rms_B": float(TB.pow(2).mean().sqrt()), "finite": bool(torch.isfinite(TA).all() and torch.isfinite(TB).all()), "shape": list(TA.shape)})
        g["latents"] = {"all_A_ne_B": all(z["max_abs_AB"] > 0 for z in zs), "cos_AB_min_mean_max": [min(z["cos_AB"] for z in zs), float(np.mean([z["cos_AB"] for z in zs])), max(z["cos_AB"] for z in zs)],
                        "rms_range": [min(min(z["rms_A"], z["rms_B"]) for z in zs), max(max(z["rms_A"], z["rms_B"]) for z in zs)], "tok_rms": S.tok_rms,
                        "all_finite_8x2560": all(z["finite"] and z["shape"] == [8, 2560] for z in zs), "online_vs_cache": LZ.get("check")}
        p = tr[0]; TA, TB = S.teacher(p["plan_A_norm"]), S.teacher(p["plan_B_norm"])
        with torch.no_grad():
            _, _, _, ax = exec_forward(S, p["question"], torch.stack([TA, TB]), [p["tgt_A"], p["tgt_A"]])
        sl = ax["slots"]; n_pre = ax["n_pre"]; pre = S.tok.decode(ax["ids"][0, :n_pre])
        nonslot = [i for i in range(ax["emb"].size(1)) if i not in sl]
        g["injection"] = {"slots_hold_ZA_ZB": bool(torch.equal(ax["emb"][0, sl], TA.to(ax["emb"].dtype)) and torch.equal(ax["emb"][1, sl], TB.to(ax["emb"].dtype))),
                          "swap_changes_only_slots": bool(torch.equal(ax["emb"][0, nonslot], ax["emb"][1, nonslot]) and torch.equal(ax["ids"][0], ax["ids"][1])
                                                          and torch.equal(ax["pos"][0], ax["pos"][1]) and torch.equal(ax["att"][0], ax["att"][1])),
                          "n_slots": len(sl), "slot_positions": sl, "slot_dim": int(ax["emb"].size(-1)),
                          "no_plan_or_exec_or_answer_in_prefix": all(t[10:80] not in pre for t in (p["plan_A_norm"], p["plan_B_norm"], p["execution_A"], p["execution_B"])),
                          "no_method_label_in_prefix": not any(lb and lb in pre for lb in (p.get("method_label_A"), p.get("method_label_B"))),
                          "prefix_head": S.tok.decode(ax["ids"][0, :6]), "prefix_tail": S.tok.decode(ax["ids"][0, n_pre - 12:n_pre]),
                          "separator_tokens": S.tok.convert_ids_to_tokens(S.T.suffix_ids), "target_tail": S.tok.decode(p["tgt_A"][-5:])}
        # CE mask: the loss is exactly the mean over target tokens
        T = p["tgt_A"]; ids1 = ax["ids"][0]; g["ce_mask"] = {"n_prefix": n_pre, "n_target": len(T), "target_ids_at_end": ids1[n_pre:n_pre + len(T)].tolist() == T,
                                                            "prefix_masked": "exec_forward slices logits [n_pre-1 : n_pre+|E|-1] -> only E + '</EXECUTION><|im_end|>' scored"}
        # forward / generation consistency on the same A/B pair
        with torch.no_grad():
            g0 = generate_batch(S, [p["question"]] * 2, torch.stack([TA, TB]), 48); g1 = generate_batch(S, [p["question"]] * 2, torch.stack([TA, TB]), 48)
        g["generation"] = {"deterministic_fresh_kv": g0 == g1, "ZA_head": g0[0][:160], "ZB_head": g0[1][:160]}
        for q_ in S.peft.parameters():
            q_.grad = None
        base_before = {n: q_.detach().clone() for n, q_ in list(S.peft.named_parameters())[:4] if "lora_" not in n}
        bridge_loss(S, p).backward()
        g["grads"] = {"lora": grad_norm(lora), "lora_B_grad": grad_norm([q_ for n, q_ in S.lora if "lora_B" in n]), "lora_A_grad": grad_norm([q_ for n, q_ in S.lora if "lora_A" in n]),
                      "projector": grad_norm([q_ for _, q_ in S.proj]),
                      "base_any_grad": any(q_.grad is not None for n, q_ in S.peft.named_parameters() if "lora_" not in n),
                      "finite": all(torch.isfinite(q_.grad).all().item() for q_ in lora if q_.grad is not None)}
        g["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        g["PASS"] = (g["requires_grad"]["projector"] == 0 and g["requires_grad"]["base"] == 0 and g["data"]["question_overlap"] == 0
                     and g["latents"]["all_A_ne_B"] and g["latents"]["all_finite_8x2560"]
                     and all(v for k, v in g["injection"].items() if isinstance(v, bool)) and g["ce_mask"]["target_ids_at_end"] and g["generation"]["deterministic_fresh_kv"]
                     and g["grads"]["lora"] > 0 and g["grads"]["projector"] == 0 and not g["grads"]["base_any_grad"] and g["grads"]["finite"]
                     and ((a.run == "A" and g["lora"]["lora_B_abs_sum"] == 0 and g["lora"]["max_abs_diff_vs_fresh"] == 0) or (a.run == "B" and g["lora"]["equals_init_ckpt"])))
        json.dump(g, open(f"{a.out}/audit.json", "w"), indent=1); print(json.dumps(g, indent=1))
        return

    if a.mode == "eval":
        tag = "step0"
        if a.ckpt:
            ck = torch.load(a.ckpt, weights_only=False); S.load(ck["state"]); tag = f"step{ck['step']}"
        S.peft.eval(); t0 = time.time(); res = {"run": a.run, "tag": tag, "ckpt": a.ckpt or (a.init or "Stage-1 + fresh LoRA")}
        rows = {"train": [], "heldout": [], "para": []}
        with torch.no_grad():
            for role, P in (("train", tr), ("heldout", ho)):
                for p in P:
                    rows[role].append({"id": short(p["id"]), **scores(S, p)})
                res[f"scores_{role}"] = summarize_scores(rows[role])
            trd = {short(p["id"]): p for p in tr}
            for r in para:
                p = trd[r["id"]]; pp = dict(p); pp["plan_A_norm"], pp["plan_B_norm"] = r["paraphrase_A"], r["paraphrase_B"]
                rows["para"].append({"id": r["id"], **scores(S, pp)})
            res["scores_para"] = summarize_scores(rows["para"]) if rows["para"] else None
        items = [(p, "Z_" + X, S.teacher(p[f"plan_{X}_norm"]), X) for p in ho for X in "AB"]
        trd = {short(p["id"]): p for p in tr}
        items += [(trd[r["id"]], "Zp_" + X, S.teacher(r["paraphrase_" + X]), X) for r in para for X in "AB"]
        if a.train_gen:
            items += [(p, "Ztr_" + X, S.teacher(p[f"plan_{X}_norm"]), X) for p in tr for X in "AB"]
        G = gen_rows(S, items, a.gen_tokens, a.gen_bs)
        gh = [r for r in G if r["latent"].startswith("Z_")]; gp = [r for r in G if r["latent"].startswith("Zp_")]; gt = [r for r in G if r["latent"].startswith("Ztr_")]
        res["gen_heldout"] = summarize_gen(gh); res["gen_para"] = summarize_gen(gp, "Zp_A", "Zp_B") if gp else None
        res["gen_train"] = summarize_gen(gt, "Ztr_A", "Ztr_B") if gt else None
        NM = normal_mode(S, normal, a.normal_tokens, 16)
        res["normal_mode"] = {"n": len(NM), "correct": int(sum(r["correct"] for r in NM)), "format_ok": int(sum(r["format_ok"] for r in NM))}
        res["eval_seconds"] = time.time() - t0; res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        os.makedirs(f"{a.out}/evaluations", exist_ok=True)
        json.dump(res, open(f"{a.out}/evaluations/eval_{tag}.json", "w"), indent=1)
        json.dump({"scores": rows, "gens": G, "normal": NM}, open(f"{a.out}/evaluations/rows_{tag}.json", "w"))
        print("EVAL", json.dumps(res), flush=True)
        return

    save = {int(x) for x in a.save_steps.split(",")}
    opt = torch.optim.AdamW(lora, lr=a.lr, weight_decay=0.0)        # fresh optimizer in both runs
    os.makedirs(f"{a.out}/checkpoints", exist_ok=True); log = open(f"{a.out}/train.jsonl", "a"); t0 = time.time()
    n_tok = sum(len(p["tgt_A"]) + len(p["tgt_B"]) for p in tr)
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True); ls = []
        for p in tr:                                                 # fixed file order, every question, A and B balanced
            l = bridge_loss(S, p); (l / len(tr)).backward(); ls.append(float(l))
        gl = grad_norm(lora); fin = all(torch.isfinite(q_.grad).all().item() for q_ in lora if q_.grad is not None)
        gn = float(torch.nn.utils.clip_grad_norm_(lora, 1.0)); opt.step()
        r = {"step": step, "train_ce": float(np.mean(ls)), "grad_lora": gl, "grad_preclip": gn, "finite": fin, "elapsed_s": time.time() - t0,
             "target_tokens_seen": n_tok * step, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
        log.write(json.dumps(r) + "\n"); log.flush(); print("STEP", json.dumps(r), flush=True)
        if not fin:
            print("STOP non-finite"); break
        if step in save:
            save_ck(S, opt, f"{a.out}/checkpoints/bridge_step{step}.pt", step, {"run": a.run, "init": a.init or "Stage-1 + fresh LoRA (seed 0)", "lr": a.lr,
                                                                                 "loss": "1.0*0.5*(CE(E_A|Q,Z_A)+CE(E_B|Q,Z_B))"})


if __name__ == "__main__":
    main()
