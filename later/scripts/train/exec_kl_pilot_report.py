"""Post-hoc analysis for the Stage-2 execution-KL pilot (forward-only + short greedy samples; no training).

Models: step-500 start, pilot A step 800, pilot B step 800. Fixed records: first 128 of stage2_val (same as probes).
Per record: execution CE (all / nonrep) for learned, zero, shuffled latents; exec KL(teacher || student) for each;
paired bootstrap CIs for gain_zero_minus_learned and gain_shuffled_minus_learned (token-weighted per record).
Five representative records: greedy continuation (first 96 execution tokens) for learned vs zero latents and the
Stage-1 teacher with the explicit plan. Also merges the training logs (probes, gradients, generation eval).
"""
import argparse
import json
import os
import random

import pandas as pd
import torch

from later.src.train.losses import stage2_exec_kl_topk_tail
from later.src.train.stage2_common import dump_json, load_stage2_model, load_tokenizer
from later.src.train.stage2_plan_dataset import PLAN_OPEN, SEPARATOR, Stage2Collator, build_views, render_prompt, stage2_tokens
from later.src.train.stage2_teacher_cache import ExecKLTeacherCache


@torch.no_grad()
def per_record(model, tok, rows, kc, L=8, bs=8):
    T = stage2_tokens(tok)
    col = Stage2Collator(tok.pad_token_id)
    recs = {}
    for s in range(0, len(rows), bs):
        chunk = rows[s: s + bs]
        b = col([build_views(tok, r, L, T, 4096) for r in chunk])
        kw = dict(input_ids=b["student_input_ids"].cuda(), attention_mask=b["student_attention_mask"].cuda(),
                  position_ids=b["student_position_ids"].cuda(), labels=b["student_labels"].cuda(),
                  boundary_positions=b["student_boundary"].cuda(), layer_ids=[36],
                  latent_positions=b["latent_positions"].cuda(), latent_start_positions=b["latent_start_positions"].cuda(),
                  return_target_logits=True)
        outs = {m: model.forward_plan_distill_stage2(**kw, latent_mode=m) for m in ("learned", "zero", "shuffled")}
        counts = outs["learned"]["target_mask"].sum(1).tolist()
        rep = b["student_seg_R"][:, 1:].cuda()[outs["learned"]["target_mask"]]
        start = 0
        for r, c in enumerate(counts):
            rid = b["record_ids"][r]
            sl = slice(start, start + int(c))
            nonrep = ~rep[sl]
            e = kc.get(rid)
            offs = torch.tensor(e["offsets"].long().tolist(), device="cuda") + start
            d = {"n_tok": int(c), "n_nonrep": int(nonrep.sum())}
            for m, o in outs.items():
                nll = o["token_nll"][sl].double()
                d[f"ce_{m}"] = float(nll.mean())
                d[f"ce_{m}_nonrep"] = float(nll[nonrep].mean())
                kl = stage2_exec_kl_topk_tail(o["target_logits"][offs], e["ids"].cuda(), e["probs"].cuda(), e["tail"].cuda())
                d[f"kl_{m}"] = float(kl.mean())
                d[f"kl16_{m}"] = float(kl[:16].mean())
            recs[rid] = d
            start += int(c)
    return recs


def boot_ci(diffs, weights, n=4000, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(diffs)))
    stat = lambda ii: sum(diffs[i] * weights[i] for i in ii) / sum(weights[i] for i in ii)
    est = stat(idx)
    bs = sorted(stat([rng.choice(idx) for _ in idx]) for _ in range(n))
    return {"estimate": est, "ci95": [bs[int(0.025 * n)], bs[int(0.975 * n)]],
            "frac_records_positive": sum(d > 0 for d in diffs) / len(diffs)}


def summarize(recs):
    ids = sorted(recs)
    w = [recs[i]["n_tok"] for i in ids]
    wn = [recs[i]["n_nonrep"] for i in ids]
    out = {}
    for m in ("learned", "zero", "shuffled"):
        out[f"ce_{m}"] = sum(recs[i][f"ce_{m}"] * w[k] for k, i in enumerate(ids)) / sum(w)
        out[f"ce_{m}_nonrep"] = sum(recs[i][f"ce_{m}_nonrep"] * wn[k] for k, i in enumerate(ids)) / sum(wn)
        out[f"exec_kl_{m}"] = sum(recs[i][f"kl_{m}"] for i in ids) / len(ids)
        out[f"first16_kl_{m}"] = sum(recs[i][f"kl16_{m}"] for i in ids) / len(ids)
    out["gain_zero_minus_learned"] = boot_ci([recs[i]["ce_zero"] - recs[i]["ce_learned"] for i in ids], w)
    out["gain_shuffled_minus_learned"] = boot_ci([recs[i]["ce_shuffled"] - recs[i]["ce_learned"] for i in ids], w)
    out["gain_zero_minus_learned_nonrep"] = boot_ci([recs[i]["ce_zero_nonrep"] - recs[i]["ce_learned_nonrep"] for i in ids], wn)
    out["kl_gain_zero_minus_learned"] = boot_ci([recs[i]["kl_zero"] - recs[i]["kl_learned"] for i in ids], [1] * len(ids))
    return out


@torch.no_grad()
def continuation(model, tok, row, mode, n_tok, explicit=False):
    T = stage2_tokens(tok)
    p = tok.encode(render_prompt(tok, row["question"]), add_special_tokens=False)
    if explicit:
        p = p + tok.encode(PLAN_OPEN + row["plan"] + SEPARATOR, add_special_tokens=False)
        g = model.generate_plan_latent([p, p], 0, [], n_tok, [tok.convert_tokens_to_ids("<|im_end|>")], tok.pad_token_id)
    else:
        p = p + T.plan_open_ids
        g = model.generate_plan_latent([p, p], 8, T.suffix_ids, n_tok, [tok.convert_tokens_to_ids("<|im_end|>")],
                                       tok.pad_token_id, latent_mode=mode)
    return tok.decode(g["token_ids"][0], skip_special_tokens=True)


def load_log(path):
    probes, train = {}, []
    for line in open(path):
        d = json.loads(line)
        if "val/ce_learned" in d or "gen/acc_learned" in d:
            probes.setdefault(d["step"], {}).update(d)
        elif "L_total" in d:
            train.append({k: d[k] for k in ("step", "L_total", "L_execution", "L_alignment", "L_explicit", "L_exec_kl",
                                            "grad_norm_latent_projector", "grad_norm_backbone", "step_time_s",
                                            "peak_allocated_GiB") if k in d})
    return probes, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--a_dir", required=True)
    ap.add_argument("--b_dir", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--kl_cache", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = load_tokenizer(args.stage1)
    rows = [pd.read_parquet(args.val).iloc[i].to_dict() for i in range(args.n)]
    kc = ExecKLTeacherCache(args.kl_cache, "val_probe", {})
    models = {"start_step500": args.start, "A_step800": os.path.join(args.a_dir, "checkpoint-800"),
              "B_step800": os.path.join(args.b_dir, "checkpoint-800")}
    res = {"n_records": len(rows), "models": models, "per_model": {}, "per_record": {}}
    for name, path in models.items():
        m = load_stage2_model(path, latent_embed_builder_no_grad=True, latent_projector_dropout=0.0,
                              dtype=torch.bfloat16).cuda().eval()
        recs = per_record(m, tok, rows, kc)
        res["per_record"][name] = recs
        res["per_model"][name] = summarize(recs)
        print(name, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res["per_model"][name].items()
                                if not isinstance(v, dict)}), flush=True)
        if name == "B_step800":
            gains = sorted(recs, key=lambda r: recs[r]["ce_zero"] - recs[r]["ce_learned"])
            pick = [gains[-1], gains[-2], gains[len(gains) // 2], gains[0], gains[1]]
            res["examples"] = []
            for rid in pick:
                row = next(r for r in rows if r["record_id"] == rid)
                res["examples"].append({"record_id": rid, "question": row["question"][:400],
                                        "gold_plan_head": row["plan"][:400],
                                        "B_metrics": recs[rid], "B_learned": continuation(m, tok, row, "learned", 96),
                                        "B_zero": continuation(m, tok, row, "zero", 96)})
        del m
        torch.cuda.empty_cache()
    m1 = load_stage2_model(args.stage1, latent_embed_builder_no_grad=True, latent_projector_dropout=0.0,
                           dtype=torch.bfloat16).cuda().eval()
    for ex in res["examples"]:
        row = next(r for r in rows if r["record_id"] == ex["record_id"])
        ex["teacher_with_plan"] = continuation(m1, tok, row, "learned", 160, explicit=True)
    for g, d in (("A", args.a_dir), ("B", args.b_dir)):
        probes, train = load_log(os.path.join(d, "train_log_rank0.jsonl"))
        res[f"log_{g}"] = {"probes": probes, "train": train,
                           "start_provenance": json.load(open(os.path.join(d, "start_provenance.json")))}
    dump_json(res, args.out)
    print("PILOT_REPORT_DONE")


if __name__ == "__main__":
    main()
