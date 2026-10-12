"""Stage 2B step-50 executor: can it read Stage-2A teacher plan latents? (0 steps, inference only)

Same frozen executor (Stage-1 Qwen3-4B + joint-2B step-50 LoRA + projector) for every condition:
  latent conditions in the 8 slots: Z_T^A, Z_T^B (frozen 2A step-300 compressor), Z_S(Q) (step-50 recurrence), zero vectors
  text oracle: chat(Q) + '<EXPLORATION>\\n' + plan text + '\\n</EXPLORATION><EXECUTION>\\n' (Stage-1 explicit mode, no latents)
Scores: mean log p over the strategy span of E (tokens after the token-LCP of E_A/E_B, before the last '\\boxed'),
teacher-forced gold prefix.  Greedy generation (2048 tokens) under every condition.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from later.scripts.bridge.bridge_input_abcd import boot
from later.scripts.bridge.mini_bridge import correct, last_boxed, method_hits, ngram_overlap
from later.scripts.stage2s.stochastic_pilot import L, build_pairs, key, short
from later.scripts.stage2t.contrastive_2b import exec_forward
from later.scripts.stage2t.three_step import TStudent, load_jsonl
from later.src.train.stage2_plan_dataset import PLAN_OPEN, SEPARATOR, render_prompt


def strategy_span(S, p, X):
    """[LCP, first token of the final '\\boxed' segment) of target X; falls back to the full E if no boxed."""
    tgt, e = p["tgt_" + X], p["execution_" + X]
    k = len(p["C"]); i = e.rfind("\\boxed")
    if i < 0:
        return (k, len(tgt) - 5), False
    n = len(S.tok.encode(e[:i], add_special_tokens=False))
    while n > 0 and tgt[:n] != S.tok.encode(e[:i], add_special_tokens=False)[:n]:
        n -= 1
    return (k, max(k + 1, n)), True


@torch.no_grad()
def text_ids(S, q, plan):
    return S.tok.encode(render_prompt(S.tok, q) + PLAN_OPEN + plan + SEPARATOR, add_special_tokens=False)


@torch.no_grad()
def text_score(S, q, plan, tgt, span):
    pre = text_ids(S, q, plan); ids = torch.tensor([pre + tgt], device=S.dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = S.base.model(input_ids=ids, position_ids=torch.arange(ids.size(1), device=S.dev)[None], use_cache=False).last_hidden_state
    lp = -F.cross_entropy(S.base.lm_head(h[0, len(pre) - 1:-1]).float(), ids[0, len(pre):], reduction="none")
    a, b = span
    return float(lp[a:b].mean())


@torch.no_grad()
def text_generate(S, q, plan, max_new):
    pre = text_ids(S, q, plan); ids = torch.tensor([pre], device=S.dev); n = len(pre)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = S.base.model(input_ids=ids, position_ids=torch.arange(n, device=S.dev)[None], use_cache=True)
        cache = out.past_key_values; lg = S.base.lm_head(out.last_hidden_state[:, -1]).float(); gen = []
        for t in range(max_new):
            tk = int(lg.argmax(-1)); gen.append(tk)
            if tk in S.eos:
                break
            out = S.base.model(input_ids=torch.tensor([[tk]], device=S.dev), position_ids=torch.tensor([[n + t]], device=S.dev), past_key_values=cache, use_cache=True)
            cache = out.past_key_values; lg = S.base.lm_head(out.last_hidden_state[:, -1]).float()
    return S.tok.decode(gen, skip_special_tokens=False)


def label(g, p):
    oa, ob = ngram_overlap(g, p["execution_A"]), ngram_overlap(g, p["execution_B"])
    return ("A" if oa > ob else "B" if ob > oa else "="), oa, ob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--s2b", required=True); ap.add_argument("--a2", required=True); ap.add_argument("--dec", required=True)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--bridge_pairs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--gen_tokens", type=int, default=2048); ap.add_argument("--n_val", type=int, default=48)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(0)
    au = {}
    ck = torch.load(a.ckpt, weights_only=False)
    S = TStudent(a.model, a.s2b, seed=0); S.load(ck["state"]); S.peft.eval()
    for p_ in S.peft.parameters():
        p_.requires_grad_(False)
    au["executor"] = {"base": a.model, "ckpt": a.ckpt, "ckpt_step": ck.get("step"), "phase": ck.get("phase"),
                      "lora_tensors_loaded": len(ck["state"]["lora"]), "projector_tensors_loaded": len(ck["state"]["proj"]),
                      "loaded_equals_ckpt": all(torch.equal(p_.detach().cpu(), ck["state"]["lora"][n].cpu()) for n, p_ in S.lora) and
                      all(torch.equal(p_.detach().cpu(), ck["state"]["proj"][n].cpu()) for n, p_ in S.proj),
                      "trainable_params": sum(p_.numel() for p_ in S.peft.parameters() if p_.requires_grad)}
    held = [p for p in build_pairs(S, a.s2b) if p["role"] == "heldout8"]
    raw = {r["id"]: r for r in load_jsonl(a.bridge_pairs)}
    singles = load_jsonl(f"{a.s2b}/data/train_singles.jsonl"); vals = load_jsonl(f"{a.s2b}/data/val_singles.jsonl")
    trq = {" ".join(s["question"].lower().split()) for s in singles}
    # ---- data check
    au["data"] = [{"id": short(p["id"]), "labels": [p.get("method_label_A"), p.get("method_label_B")], "verified": [raw[p["id"]]["verified_A"], raw[p["id"]]["verified_B"]],
                   "pred": [raw[p["id"]]["pred_A"], raw[p["id"]]["pred_B"]], "gold": p["answer"],
                   "pred_correct": [correct(raw[p["id"]]["pred_A"], str(p["answer"])), correct(raw[p["id"]]["pred_B"], str(p["answer"]))],
                   "in_train_singles": " ".join(p["question"].lower().split()) in trq} for p in held]
    # ---- encoder consistency: online 2A step-300 compressor vs the cached Z_T used in 2B
    from later.scripts.stage2a.plan_ae import System
    R = System(a.dec, f"{a.a2}/data", seed=42); R.load(torch.load(f"{a.a2}/checkpoints/step300.pt", weights_only=False)["state"])
    texts = [p[k] for p in held for k in ("plan_A_norm", "plan_B_norm")]
    have = [t for t in texts if key(t) in R.states]
    au["encoder_states_available"] = f"{len(have)}/{len(texts)}"
    if have:
        with torch.no_grad():
            Zon = R.Z(have)
        Zc = torch.stack([S.Zraw[key(t)].float().to(S.dev) for t in have])
        au["encoder_online_vs_cache_maxabs"] = float((Zon - Zc).abs().max())
    else:
        au["encoder_online_vs_cache_maxabs"] = None
    au["encoder_ckpt"] = f"{a.a2}/checkpoints/step300.pt"
    del R; torch.cuda.empty_cache()
    # ---- latents
    with torch.no_grad():
        ZS48 = torch.stack([S.zlat(v["question"]) for v in vals[:a.n_val]]); T48 = torch.stack([S.teacher(v["plan_norm"]) for v in vals[:a.n_val]])
    fl = ZS48.flatten(1); cc = F.cosine_similarity(fl[:, None], fl[None], -1); iu = torch.triu_indices(len(fl), len(fl), 1)
    au["student_collapse_48val"] = {"crossq_cos": float(cc[iu[0], iu[1]].mean()), "var_over_teacher": float(ZS48.var(0).mean() / T48.var(0).mean())}
    rows = []; slotstats = {"TA": [], "TB": [], "ZS": []}; inj = []
    for p in held:
        q = p["question"]
        with torch.no_grad():
            TA, TB = S.teacher(p["plan_A_norm"]), S.teacher(p["plan_B_norm"]); ZS = S.zlat(q); Z0 = torch.zeros_like(ZS)
        for nm, Z in (("TA", TA), ("TB", TB), ("ZS", ZS)):
            slotstats[nm].append(torch.stack([Z.norm(dim=-1), Z.mean(-1), Z.var(-1)], -1))      # [8,3]
        spA, okA = strategy_span(S, p, "A"); spB, okB = strategy_span(S, p, "B")
        conds = [("TA", TA), ("TB", TB), ("ZS", ZS), ("zero", Z0)]
        with torch.no_grad():
            Zall = torch.stack([z for _, z in conds])
            _, sA, _, axA = exec_forward(S, q, Zall, [p["tgt_A"]] * 4, [spA] * 4)
            _, sB, _, axB = exec_forward(S, q, Zall, [p["tgt_B"]] * 4, [spB] * 4)
            sfA, _, _, _ = exec_forward(S, q, Zall, [p["tgt_A"]] * 4)
            sfB, _, _, _ = exec_forward(S, q, Zall, [p["tgt_B"]] * 4)
        sl = axA["slots"]; e = axA["emb"]; ns = torch.ones(e.size(1), dtype=torch.bool); ns[sl] = False
        inj.append({"id": short(p["id"]), "slots_hold_latents": all(torch.equal(e[i, sl], conds[i][1].to(e.dtype)) for i in range(4)),
                    "nonslot_emb_identical": all(torch.equal(e[0, ns], e[i, ns]) for i in range(4)),
                    "ids_pos_att_identical": bool((axA["ids"] == axA["ids"][0]).all() and (axA["pos"] == axA["pos"][0]).all() and (axA["att"] == axA["att"][0]).all()),
                    "no_plan_text_in_prefix": all(pl[10:80] not in S.tok.decode(axA["ids"][0, :axA["n_pre"]]) for pl in (p["plan_A_norm"], p["plan_B_norm"], raw[p["id"]]["plan_A"], raw[p["id"]]["plan_B"])),
                    "prefix_tail": S.tok.decode(axA["ids"][0, axA["n_pre"] - 12:axA["n_pre"]])})
        sc = {f"s_{c}_E{X}": float(v) for X, s in (("A", sA), ("B", sB)) for c, v in zip([c for c, _ in conds], s)}
        fu = {f"full_{c}_E{X}": float(v) for X, s in (("A", sfA), ("B", sfB)) for c, v in zip([c for c, _ in conds], s)}
        r = {"id": short(p["id"]), "label_A": p.get("method_label_A"), "label_B": p.get("method_label_B"), "span_A": spA, "span_B": spB, "span_excludes_boxed": okA and okB,
             **sc, **fu}
        r["s_AA"], r["s_AB"], r["s_BA"], r["s_BB"] = sc["s_TA_EA"], sc["s_TA_EB"], sc["s_TB_EA"], sc["s_TB_EB"]
        r["delta_A"] = r["s_AA"] - r["s_BA"]; r["delta_B"] = r["s_BB"] - r["s_AB"]; r["delta_match"] = (r["delta_A"] + r["delta_B"]) / 2
        r["full_delta_match"] = ((fu["full_TA_EA"] - fu["full_TB_EA"]) + (fu["full_TB_EB"] - fu["full_TA_EB"])) / 2
        r["dnll_zero_minus_student"] = ((sc["s_ZS_EA"] - sc["s_zero_EA"]) + (sc["s_ZS_EB"] - sc["s_zero_EB"])) / 2      # NLL(zero) - NLL(student)
        r["teacher_AB_cos"] = float(F.cosine_similarity(TA.flatten(), TB.flatten(), 0)); r["teacher_AB_l2_rel"] = float((TA - TB).norm() / TA.norm())
        r["ZS_vs_TA_cos"] = float(F.cosine_similarity(ZS.flatten(), TA.flatten(), 0)); r["ZS_vs_TB_cos"] = float(F.cosine_similarity(ZS.flatten(), TB.flatten(), 0))
        # text oracle on the same executor
        pa, pb = raw[p["id"]]["plan_A"], raw[p["id"]]["plan_B"]
        r["oracle_s_AA"], r["oracle_s_AB"] = text_score(S, q, pa, p["tgt_A"], spA), text_score(S, q, pa, p["tgt_B"], spB)
        r["oracle_s_BA"], r["oracle_s_BB"] = text_score(S, q, pb, p["tgt_A"], spA), text_score(S, q, pb, p["tgt_B"], spB)
        r["oracle_delta_match"] = ((r["oracle_s_AA"] - r["oracle_s_BA"]) + (r["oracle_s_BB"] - r["oracle_s_AB"])) / 2
        # generation
        gens, _ = S.generate(q, Zall, a.gen_tokens)
        gens = dict(zip([c for c, _ in conds], gens)); gens["text_PA"] = text_generate(S, q, pa, a.gen_tokens); gens["text_PB"] = text_generate(S, q, pb, a.gen_tokens)
        r["gen"] = {}
        for c, g in gens.items():
            lb, oa, ob = label(g, p); pr = last_boxed(g)
            r["gen"][c] = {"label": lb, "ov_A": oa, "ov_B": ob, "terms_A": method_hits(g, p["method_terms_only_A"]), "terms_B": method_hits(g, p["method_terms_only_B"]),
                           "pred": pr, "correct": correct(pr, str(p["answer"])), "format_ok": "</EXECUTION>" in g and "<|im_end|>" in g, "text": g}
        r["teacher_switch"] = r["gen"]["TA"]["label"] == "A" and r["gen"]["TB"]["label"] == "B"
        r["text_switch"] = r["gen"]["text_PA"]["label"] == "A" and r["gen"]["text_PB"]["label"] == "B"
        r["TA_TB_identical_text"] = gens["TA"] == gens["TB"]
        rows.append(r); print(r["id"], round(r["delta_match"], 5), r["teacher_switch"], r["text_switch"], round(r["oracle_delta_match"], 4), flush=True)
    for nm in slotstats:
        x = torch.stack(slotstats[nm]).mean(0)
        au[f"slot_stats_{nm}_norm_mean_var"] = [[float(v) for v in s] for s in x]
    au["injection"] = inj
    au["INTERFACE_PASS"] = au["executor"]["loaded_equals_ckpt"] and au["executor"]["trainable_params"] == 0 and (au["encoder_online_vs_cache_maxabs"] is not None and au["encoder_online_vs_cache_maxabs"] < 1e-3) and \
        all(i["slots_hold_latents"] and i["nonslot_emb_identical"] and i["ids_pos_att_identical"] and i["no_plan_text_in_prefix"] for i in inj)
    st = lambda v: {"mean": float(np.mean(v)), "ci95": boot(list(map(float, v))), "n_pos": int(sum(x > 0 for x in v)), "n": len(v)}
    summ = {"teacher_delta_match": st([r["delta_match"] for r in rows]), "teacher_delta_A": st([r["delta_A"] for r in rows]), "teacher_delta_B": st([r["delta_B"] for r in rows]),
            "teacher_full_delta_match": st([r["full_delta_match"] for r in rows]), "oracle_delta_match": st([r["oracle_delta_match"] for r in rows]),
            "dnll_zero_minus_student": st([r["dnll_zero_minus_student"] for r in rows]),
            "teacher_switch": int(sum(r["teacher_switch"] for r in rows)), "text_switch": int(sum(r["text_switch"] for r in rows)),
            "TA_TB_identical_text": int(sum(r["TA_TB_identical_text"] for r in rows)),
            "labels": {c: "".join(r["gen"][c]["label"] for r in rows) for c in ("TA", "TB", "ZS", "zero", "text_PA", "text_PB")},
            "correct": {c: int(sum(r["gen"][c]["correct"] for r in rows)) for c in ("TA", "TB", "ZS", "zero", "text_PA", "text_PB")},
            "format_ok": {c: int(sum(r["gen"][c]["format_ok"] for r in rows)) for c in ("TA", "TB", "ZS", "zero", "text_PA", "text_PB")},
            "teacher_AB_cos_mean": float(np.mean([r["teacher_AB_cos"] for r in rows])), "teacher_AB_l2_rel_mean": float(np.mean([r["teacher_AB_l2_rel"] for r in rows]))}
    json.dump(au, open(f"{a.out}/audit.json", "w"), indent=1); json.dump(summ, open(f"{a.out}/summary.json", "w"), indent=1)
    with open(f"{a.out}/rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    keys = [k for k in rows[0] if k != "gen"]
    with open(f"{a.out}/per_question.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(keys + [f"gen_{c}_{x}" for c in ("TA", "TB", "ZS", "zero", "text_PA", "text_PB") for x in ("label", "correct", "format_ok")])
        for r in rows:
            w.writerow([json.dumps(r[k]) if isinstance(r[k], (list, tuple)) else r[k] for k in keys] + [r["gen"][c][x] for c in ("TA", "TB", "ZS", "zero", "text_PA", "text_PB") for x in ("label", "correct", "format_ok")])
    print(json.dumps({"INTERFACE_PASS": au["INTERFACE_PASS"], **summ}, indent=1))


if __name__ == "__main__":
    main()
