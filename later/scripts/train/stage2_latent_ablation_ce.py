"""Forward-only latent-usage diagnostic (teacher forcing on held-out val records).

Execution CE (E + A + closing tokens, token-weighted) of each model under:
  gold_plan     explicit path Q -> P_text -> [E]            (upper reference: full plan information)
  learned       latent path with the model's own latents
  zero          same L positions, latent embeddings set to 0
  no_latent     L = 0, separator right after '<EXPLORATION>\\n'
If learned ≈ zero ≈ no_latent, the latents carry no usable plan information.
"""
import argparse
import json
import os

import pandas as pd
import torch

from later.src.train.stage2_common import dump_json, load_stage2_model, load_tokenizer
from later.src.train.stage2_plan_dataset import Stage2Collator, build_views, stage2_tokens


def exec_ce(model, batch, view, latent_mode="learned", layer_ids=(1,)):
    dev = "cuda"
    labels = batch[f"{view}_labels"].clone()
    if view == "explicit":  # keep only execution-side targets (E, A, closing) for a like-for-like comparison
        keep = batch["explicit_seg_E"] | batch["explicit_seg_A"] | batch["explicit_seg_C"]
        labels[~keep] = -100
    kw = dict(input_ids=batch[f"{view}_input_ids"].to(dev), attention_mask=batch[f"{view}_attention_mask"].to(dev),
              position_ids=batch[f"{view}_position_ids"].to(dev), labels=labels.to(dev),
              boundary_positions=batch[f"{view}_boundary"].to(dev), layer_ids=list(layer_ids))
    if view == "student":
        kw.update(latent_positions=batch["latent_positions"].to(dev),
                  latent_start_positions=batch["latent_start_positions"].to(dev), latent_mode=latent_mode)
    out = model.forward_plan_distill_stage2(**kw)
    return float(out["token_nll"].sum()), int(out["token_nll"].numel()), out["builder_stats"], out["boundary_hidden"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path")
    ap.add_argument("--val", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--latent_steps", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache_dir", default=None, help="teacher cache -> also report alignment loss per condition")
    args = ap.parse_args()
    tok = load_tokenizer(args.models[0].split("=", 1)[1])
    T = stage2_tokens(tok)
    frame = pd.read_parquet(args.val).iloc[: args.n]
    collate = Stage2Collator(tok.pad_token_id)
    cache = None
    if args.cache_dir:
        from later.src.train.losses import stage2_normalized_l1_alignment
        from later.src.train.stage2_teacher_cache import TeacherBoundaryCache
        cache = TeacherBoundaryCache(args.cache_dir, "val", {})
    layers = list(range(1, 37))
    res = {"n_records": len(frame), "latent_steps": args.latent_steps, "models": {}}
    for spec in args.models:
        name, path = spec.split("=", 1)
        m = load_stage2_model(path, latent_embed_builder_no_grad=True, latent_projector_dropout=0.0,
                              dtype=torch.bfloat16).cuda().eval()
        acc = {k: [0.0, 0] for k in ("gold_plan", "learned", "zero", "no_latent")}
        align = {k: 0.0 for k in ("gold_plan", "learned", "zero", "no_latent")}
        norms = []
        with torch.no_grad():
            for i in range(len(frame)):
                r = frame.iloc[i].to_dict()
                b = collate([build_views(tok, r, args.latent_steps, T, 4096)])
                b0 = collate([build_views(tok, r, 0, T, 4096)])
                for key, (bb, view, mode) in {"gold_plan": (b, "explicit", "learned"), "learned": (b, "student", "learned"),
                                              "zero": (b, "student", "zero"), "no_latent": (b0, "student", "learned")}.items():
                    s, n, st, bh = exec_ce(m, bb, view, mode, layers if cache else (1,))
                    acc[key][0] += s
                    acc[key][1] += n
                    if cache is not None:
                        th = cache.get(bb["record_ids"], layers, "cuda")
                        align[key] += float(stage2_normalized_l1_alignment(bh, th, layers)[0]) / len(frame)
                    if key == "learned":
                        norms.append((st["latent_embed_norm_mean"], st["token_embed_norm_mean"]))
        ce = {k: v[0] / v[1] for k, v in acc.items()}
        span = ce["no_latent"] - ce["gold_plan"]
        res["models"][name] = {"exec_ce": ce, "n_exec_tokens": acc["learned"][1],
                               "plan_info_recovered_learned": (ce["no_latent"] - ce["learned"]) / span if span else None,
                               "latent_embed_norm_mean": sum(a for a, _ in norms) / len(norms),
                               "token_embed_norm_mean": sum(b for _, b in norms) / len(norms),
                               "alignment_loss": align if cache is not None else None}
        print(name, json.dumps(res["models"][name]), flush=True)
        del m
        torch.cuda.empty_cache()
    dump_json(res, args.out)


if __name__ == "__main__":
    main()
