"""Test 2 (diagnostics, forward-only): does PlanRec / execution depend on the CORRECT question's latents?

For a fixed set of val records (first N, same as the pilot probes) and one model, with the reference plan P, the
PlanRec target mask and the execution targets held fixed, only the latents fed to the branches change:
  learned | shuffled_s{0,1,2} (3 fixed derangements over the N records) | mean (per-slot mean over the N records) | zero
Per record: PlanRec CE (all, first 16 / first 32 plan targets, non-template content tokens, first 32 content tokens),
execution CE, KL(learned || cond) of the PlanRec next-token distributions on the first 16 plan targets. Paired bootstrap
over records (2,000 resamples, seed 0), token-weighted and per-record mean. Latent RMS / per-slot cosine / variance.
Free plan reconstruction from Z alone for the first K records under learned / shuffled_s0 / mean / zero.
Implementation checks on this model: future-plan-token invariance, Z invariant to E/A edits, swapped Z reaches both
branches (tensor hash + output change). model.eval() + torch.inference_mode(); no parameter is modified.
"""
import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter

import pandas as pd
import torch

from later.src.train.stage2_common import dump_json, load_stage2_model, load_tokenizer
from later.src.train.stage2_plan_dataset import (EXEC_CLOSE, PLAN_CONTENT_RULE, PLAN_OPEN, SEPARATOR, Stage2Collator,
                                                 build_views, stage2_tokens)
from later.src.train.stage2_planrec_monitor import build_latents, fixed_derangement

COND = ["learned", "shuffled_s0", "shuffled_s1", "shuffled_s2", "mean", "zero"]


def thash(t):
    return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]


def boot(diffs, weights, n=2000, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(diffs)))
    stat = lambda ii: sum(diffs[i] * weights[i] for i in ii) / max(sum(weights[i] for i in ii), 1e-12)
    bs = sorted(stat([rng.choice(idx) for _ in idx]) for _ in range(n))
    return {"estimate": stat(idx), "ci95": [bs[int(0.025 * n)], bs[int(0.975 * n) - 1]],
            "frac_records_positive": sum(d > 0 for d in diffs) / len(diffs), "n_records": len(diffs)}


def token_f1(a, b):
    ta, tb = re.findall(r"\w+|\S", a.lower()), re.findall(r"\w+|\S", b.lower())
    ov = sum((Counter(ta) & Counter(tb)).values())
    if not ta or not tb or ov == 0:
        return 0.0
    p, r = ov / len(ta), ov / len(tb)
    return 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--n_free", type=int, default=20)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = load_tokenizer(a.stage1)
    T = stage2_tokens(tok)
    model = load_stage2_model(a.model, latent_embed_builder_no_grad=True, latent_projector_dropout=0.0,
                              dtype=torch.float32, latent_output_rms="token_embedding").cuda().eval()
    w_hash = thash(torch.cat([p.flatten() for p in model.latent_projector.parameters()]))
    val = pd.read_parquet(a.val)
    rows = [val.iloc[i].to_dict() for i in range(a.n)]
    col = Stage2Collator(tok.pad_token_id)
    batches = [col([build_views(tok, r, 8, T, 4096) for r in rows[s: s + a.bs]]) for s in range(0, a.n, a.bs)]
    dev = "cuda"
    ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        Z = torch.cat([build_latents(model, b) for b in batches])            # [N, 8, d] fp32
        perms = {f"shuffled_s{s}": fixed_derangement(a.n, seed=s) for s in range(3)}
        zmean = Z.mean(0)

        def zc(cond, idx):
            if cond == "learned":
                return Z[idx]
            if cond == "zero":
                return torch.zeros_like(Z[idx])
            if cond == "mean":
                return zmean.unsqueeze(0).expand(len(idx), -1, -1)
            return Z[[perms[cond][i] for i in idx]]

        recs = {}
        off = 0
        for b in batches:
            B = b["student_input_ids"].size(0)
            idx = list(range(off, off + B))
            off += B
            lp_learned = None
            for cond in COND:
                z = zc(cond, idx)
                with ac():
                    lat = model.forward_plan_distill_stage2(
                        input_ids=b["student_input_ids"].to(dev), attention_mask=b["student_attention_mask"].to(dev),
                        position_ids=b["student_position_ids"].to(dev), labels=b["student_labels"].to(dev),
                        boundary_positions=b["student_boundary"].to(dev), layer_ids=[36],
                        latent_positions=b["latent_positions"].to(dev),
                        latent_start_positions=b["latent_start_positions"].to(dev), latent_override=z)
                    rec = model.forward_plan_reconstruction(z, b["planrec_input_ids"].to(dev),
                                                            b["planrec_attention_mask"].to(dev),
                                                            b["planrec_labels"].to(dev), return_logits=True)
                logp = torch.log_softmax(rec["target_logits"].float(), -1)
                ec, rc = lat["target_mask"].sum(1).tolist(), rec["target_mask"].sum(1).tolist()
                cont = b["planrec_content"][:, 1:].to(dev)[rec["target_mask"]]
                es = rs = 0
                per_lp = []
                for i in range(B):
                    rid = b["record_ids"][i]
                    en = lat["token_nll"][es: es + ec[i]].double()
                    rn = rec["token_nll"][rs: rs + rc[i]].double()
                    cm = cont[rs: rs + rc[i]]
                    cn = rn[cm]
                    lp16 = logp[rs: rs + min(16, rc[i])]
                    per_lp.append(lp16)
                    d = {"exec_sum": float(en.sum()), "exec_n": en.numel(), "rec_sum": float(rn.sum()), "rec_n": rn.numel(),
                         "first16_sum": float(rn[:16].sum()), "first16_n": min(16, rn.numel()),
                         "first32_sum": float(rn[:32].sum()), "first32_n": min(32, rn.numel()),
                         "content_sum": float(cn.sum()), "content_n": cn.numel(),
                         "content32_sum": float(cn[:32].sum()), "content32_n": min(32, cn.numel()),
                         "first_token_nll": float(rn[0]) if rn.numel() else None}
                    recs.setdefault(rid, {})[cond] = d
                    es += ec[i]; rs += rc[i]
                if cond == "learned":
                    lp_learned = per_lp
                else:   # KL(learned || cond) on the first 16 plan targets, mean over positions
                    for i in range(B):
                        p, q = lp_learned[i], per_lp[i]
                        recs[b["record_ids"][i]][cond]["kl16_from_learned"] = float((p.exp() * (p - q)).sum(-1).mean())
        # ---- latent geometry ----
        N = Z.size(0)
        off_diag = ~torch.eye(N, dtype=torch.bool, device=Z.device)
        nz = torch.nn.functional.normalize
        per_slot = [float((nz(Z[:, k], dim=-1) @ nz(Z[:, k], dim=-1).T)[off_diag].mean()) for k in range(8)]
        flat = Z.reshape(N, -1)
        tok_rms = float(model.get_input_embeddings().weight.float().pow(2).mean().sqrt())
        geom = {"cosine_flat": float((nz(flat, dim=-1) @ nz(flat, dim=-1).T)[off_diag].mean()),
                "cosine_per_slot": per_slot,
                "cosine_flat_centered": float((nz(flat - flat.mean(0), dim=-1) @ nz(flat - flat.mean(0), dim=-1).T)[off_diag].mean()),
                "cosine_matched_vs_shuffled_note": "shuffled latents are other questions' Z; cosine to learned = cosine_flat on average",
                "rms_per_slot_over_token_rms": [float(Z[:, k].pow(2).mean().sqrt()) / tok_rms for k in range(8)],
                "between_question_variance_fraction_per_slot": [float(Z[:, k].var(0).sum() / Z[:, k].pow(2).sum(-1).mean()) for k in range(8)],
                "token_embedding_rms": tok_rms}
        # ---- free reconstruction ----
        stop = [tok.encode("</", add_special_tokens=False)[0], tok.convert_tokens_to_ids("<|im_end|>")]
        K = min(a.n_free, N)
        idx = list(range(K))
        free = {}
        first_tok = {}
        for cond in ("learned", "shuffled_s0", "mean", "zero"):
            z = zc(cond, idx).cuda()
            with ac():
                ids = model.generate_plan_reconstruction(z, T.plan_open_ids, 320, stop)
                # next-token distribution at the first plan position (no plan prefix fed)
                x = torch.cat([z.to(model.dtype), model.get_input_embeddings()(torch.tensor(T.plan_open_ids, device=dev)).unsqueeze(0).expand(K, -1, -1)], 1)
                h = model.model(inputs_embeds=x, position_ids=torch.arange(x.size(1), device=dev)[None].expand(K, -1)).last_hidden_state[:, -1]
                first_tok[cond] = torch.log_softmax(model.lm_head(h).float(), -1)
            free[cond] = [(tok.decode(g, skip_special_tokens=True), bool(g and g[-1] in stop), len(g)) for g in ids]
        golds = [r["plan"] for r in rows[:K]]
        free_rows, fsum = [], {}
        for cond, outs in free.items():
            texts = [t.split("</")[0] for t, _, _ in outs]
            ident = 0
            for i, t in enumerate(texts):
                f = [token_f1(t, g) for g in golds]
                ident += max(range(K), key=f.__getitem__) == i
            lines = [l.strip() for t in texts for l in t.split("\n") if l.strip()]
            fsum[cond] = {"distinct_outputs": len(set(texts)), "n": K, "identifies_own_plan_among_K": ident,
                          "mean_f1_own_plan": sum(token_f1(t, g) for t, g in zip(texts, golds)) / K,
                          "end_rate": sum(s for _, s, _ in outs) / K, "mean_len_tokens": sum(n for _, _, n in outs) / K,
                          "repeated_line_fraction": 1 - len(set(lines)) / max(1, len(lines))}
            for i, (t, s, n) in enumerate(outs):
                if cond == "learned":
                    free_rows.append({"record_id": rows[i]["record_id"], "question": rows[i]["question"], "gold_plan": golds[i]})
                fr = free_rows[i]
                fr[f"rec_{cond}"] = texts[i]; fr[f"stopped_{cond}"] = s; fr[f"len_{cond}"] = n
                fr[f"f1_own_{cond}"] = token_f1(texts[i], golds[i])
        kl0 = {c: float((first_tok["learned"].exp() * (first_tok["learned"] - first_tok[c])).sum(-1).mean())
               for c in ("shuffled_s0", "mean", "zero")}
        top1 = {c: [tok.decode([int(i)]) for i in first_tok[c].argmax(-1).tolist()] for c in first_tok}
        # ---- implementation checks on THIS model ----
        checks = {}
        b = batches[0]
        z = Z[:b["student_input_ids"].size(0)]
        ids = b["planrec_input_ids"].to(dev)
        lab = b["planrec_labels"].to(dev)
        msk = b["planrec_attention_mask"].to(dev)
        o1 = model.forward_plan_reconstruction(z, ids, msk, lab, return_logits=True)
        first = 8 + len(T.plan_open_ids)
        k = first + 40
        ids2, lab2 = ids.clone(), lab.clone()
        ids2[0, k] = (ids2[0, k] + 11) % 1000
        lab2[0, k] = ids2[0, k]
        o2 = model.forward_plan_reconstruction(z, ids2, msk, lab2, return_logits=True)
        n0 = int(o1["target_mask"][0].sum())
        j = k - first
        checks["future_plan_token_invariance"] = {
            "perturbed_target_index": j, "max_abs_logit_diff_before": float((o1["target_logits"][:j] - o2["target_logits"][:j]).abs().max()),
            "max_abs_logit_diff_after": float((o1["target_logits"][j + 1:n0] - o2["target_logits"][j + 1:n0]).abs().max())}
        checks["future_plan_token_invariance"]["pass"] = checks["future_plan_token_invariance"]["max_abs_logit_diff_before"] < 1e-3 \
            and checks["future_plan_token_invariance"]["max_abs_logit_diff_after"] > 1e-3
        r = dict(rows[0])
        r2 = dict(r, execution="A different execution. The answer is $\\boxed{7}$.")
        r2["answer_char_start_in_execution"] = r2["execution"].index("\\boxed")
        r2["stage1_solution"] = PLAN_OPEN + r2["plan"] + SEPARATOR + r2["execution"] + EXEC_CLOSE
        za = build_latents(model, col([build_views(tok, r, 8, T, 4096)]))
        zb = build_latents(model, col([build_views(tok, r2, 8, T, 4096)]))
        checks["z_invariant_to_execution_answer"] = {"hash_original": thash(za), "hash_edited_EA": thash(zb), "pass": thash(za) == thash(zb)}
        zs = Z[[perms["shuffled_s0"][i] for i in range(z.size(0))]]
        o3 = model.forward_plan_reconstruction(zs, ids, msk, lab, return_logits=True)
        lat_l = model.forward_plan_distill_stage2(input_ids=b["student_input_ids"].to(dev), attention_mask=b["student_attention_mask"].to(dev),
                                                  position_ids=b["student_position_ids"].to(dev), labels=b["student_labels"].to(dev),
                                                  boundary_positions=b["student_boundary"].to(dev), layer_ids=[36],
                                                  latent_positions=b["latent_positions"].to(dev),
                                                  latent_start_positions=b["latent_start_positions"].to(dev), latent_override=z)
        lat_s = model.forward_plan_distill_stage2(input_ids=b["student_input_ids"].to(dev), attention_mask=b["student_attention_mask"].to(dev),
                                                  position_ids=b["student_position_ids"].to(dev), labels=b["student_labels"].to(dev),
                                                  boundary_positions=b["student_boundary"].to(dev), layer_ids=[36],
                                                  latent_positions=b["latent_positions"].to(dev),
                                                  latent_start_positions=b["latent_start_positions"].to(dev), latent_override=zs)
        rows_ix = torch.arange(z.size(0), device=dev).unsqueeze(1)
        ins = lat_s["inputs_embeds"][rows_ix, b["latent_positions"].to(dev)]
        checks["swapped_z_reaches_branches"] = {
            "exec_inserted_hash_equals_swapped_z": thash(ins) == thash(zs.to(ins.dtype)),
            "exec_inserted_hash_differs_from_learned": thash(ins) != thash(z.to(ins.dtype)),
            "planrec_logits_max_abs_change": float((o1["target_logits"] - o3["target_logits"]).abs().max()),
            "exec_nll_max_abs_change": float((lat_l["token_nll"] - lat_s["token_nll"]).abs().max())}
        c = checks["swapped_z_reaches_branches"]
        c["pass"] = c["exec_inserted_hash_equals_swapped_z"] and c["exec_inserted_hash_differs_from_learned"] \
            and c["planrec_logits_max_abs_change"] > 1e-3 and c["exec_nll_max_abs_change"] > 1e-4
        checks["planrec_context_has_no_question_tokens"] = {
            "pass": all(tok.decode(v[8:]) == PLAN_OPEN + rr["plan"] + "\n" for v, rr in
                        ((build_views(tok, rr, 8, T, 4096)["planrec_ids"], rr) for rr in rows[:32]))}
        checks["note"] = "checks run in fp32 without autocast on batch 0 of this model"
    # ---- metrics ----
    ids_sorted = sorted(recs)
    res = {"name": a.name, "model": a.model, "projector_hash": w_hash, "n_records": len(recs), "record_ids": [r["record_id"] for r in rows],
           "content_rule": PLAN_CONTENT_RULE, "bootstrap": {"n": 2000, "seed": 0, "unit": "record (paired)"},
           "conditions": {}, "gains": {}, "geometry": geom, "free_reconstruction": fsum,
           "first_plan_token_kl_from_learned_no_prefix": kl0,
           "first_plan_token_top1": {c: Counter(v).most_common(5) for c, v in top1.items()}, "implementation_checks": checks}
    keys = ["exec", "rec", "first16", "first32", "content", "content32"]
    tot_rec = sum(recs[i]["learned"]["rec_n"] for i in ids_sorted)
    res["content_coverage"] = sum(recs[i]["learned"]["content_n"] for i in ids_sorted) / tot_rec
    for cond in COND:
        d = {}
        for k in keys:
            d[f"{k}_ce"] = sum(recs[i][cond][f"{k}_sum"] for i in ids_sorted) / sum(recs[i][cond][f"{k}_n"] for i in ids_sorted)
            vals = [recs[i][cond][f"{k}_sum"] / recs[i][cond][f"{k}_n"] for i in ids_sorted if recs[i][cond][f"{k}_n"]]
            d[f"{k}_ce_per_record"] = sum(vals) / len(vals)
        if cond != "learned":
            d["kl16_from_learned"] = sum(recs[i][cond]["kl16_from_learned"] for i in ids_sorted) / len(ids_sorted)
        res["conditions"][cond] = d
    for cond in COND[1:]:
        for k in keys:
            diffs_tw = [(recs[i][cond][f"{k}_sum"] - recs[i]["learned"][f"{k}_sum"]) / max(recs[i]["learned"][f"{k}_n"], 1) for i in ids_sorted]
            w = [recs[i]["learned"][f"{k}_n"] for i in ids_sorted]
            res["gains"][f"{k}_gain_{cond}"] = boot(diffs_tw, w)
            res["gains"][f"{k}_gain_{cond}_per_record"] = boot(diffs_tw, [1.0 if x else 0.0 for x in w])
    # shuffled pooled over the three seeds (per record: mean of the three gains)
    for k in keys:
        diffs = [sum((recs[i][f"shuffled_s{s}"][f"{k}_sum"] - recs[i]["learned"][f"{k}_sum"]) for s in range(3)) / 3 / max(recs[i]["learned"][f"{k}_n"], 1)
                 for i in ids_sorted]
        res["gains"][f"{k}_gain_shuffled_3seed_mean"] = boot(diffs, [recs[i]["learned"][f"{k}_n"] for i in ids_sorted])
    dump_json(res, os.path.join(a.out, f"metrics_{a.name}.json"))
    out_rows = []
    for i in ids_sorted:
        for cond in COND:
            d = recs[i][cond]
            out_rows.append({"model": a.name, "record_id": i, "condition": cond,
                             **{f"{k}_ce": (d[f"{k}_sum"] / d[f"{k}_n"]) if d[f"{k}_n"] else None for k in keys},
                             **{f"{k}_tokens": d[f"{k}_n"] for k in keys}, "first_token_nll": d["first_token_nll"],
                             "kl16_from_learned": d.get("kl16_from_learned")})
    pd.DataFrame(out_rows).to_csv(os.path.join(a.out, f"paired_per_example_{a.name}.csv"), index=False)
    with open(os.path.join(a.out, f"free_planrec_samples_{a.name}.jsonl"), "w") as f:
        for r in free_rows:
            f.write(json.dumps({"model": a.name, **r}, ensure_ascii=False) + "\n")
    print("DIAG2_DONE", a.name, json.dumps({c: round(res["conditions"][c]["rec_ce"], 5) for c in COND}), json.dumps(checks, default=str)[:600])


if __name__ == "__main__":
    main()
