"""Offline Stage-2 teacher targets: frozen Stage-1 checkpoint, prefix-only forward (Q + P + boundary), decoder-layer
outputs at the boundary token for every alignment layer. Not related to LaTER's PrecomputedTeacherCache (top-k
logits for KL); this stores hidden-state targets only.

Output: <out>/<split>_shard{k}.pt  {"record_ids": [...], "hidden": bf16[N, n_layers, d], "boundary_pos": [...]}
        <out>/manifest.json (checkpoint/tokenizer/template hashes, boundary rule, layer ids, hidden dim, record ids)
"""
import argparse
import os
import sys
import time

import torch

from later.src.train.stage2_common import (checkpoint_hash, dump_json, load_stage2_model, load_tokenizer,
                                           tokenizer_hashes)
from later.src.train.stage2_plan_dataset import BOUNDARY_RULE_VERSION, Stage2Collator, Stage2PlanDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--data", nargs="+", required=True, help="split=path.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--shard_size", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok = load_tokenizer(args.teacher)
    model = load_stage2_model(args.teacher, latent_embed_builder_no_grad=True, latent_projector_dropout=0.0,
                              dtype=torch.bfloat16).cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layer_ids = model.resolve_alignment_layers(args.layers, model.config.num_hidden_layers)
    manifest = {"teacher_checkpoint": os.path.abspath(args.teacher), "teacher_checkpoint_hash": checkpoint_hash(args.teacher),
                **tokenizer_hashes(tok, args.teacher), "boundary_rule_version": BOUNDARY_RULE_VERSION,
                "layer_ids": layer_ids, "layer_semantics": "decoder layer output (residual stream) before final norm, 1-based",
                "hidden_dim": model.config.hidden_size, "dtype": "bfloat16", "splits": {}}
    collate = Stage2Collator(pad_id=tok.pad_token_id)
    for spec in args.data:
        split, path = spec.split("=", 1)
        ds = Stage2PlanDataset(path, tok, latent_steps=1, limit=args.limit or None)
        shards, rids, hidden, bpos, t0 = [], [], [], [], time.time()

        def flush():
            if not rids:
                return
            name = f"{split}_shard{len(shards)}.pt"
            torch.save({"record_ids": list(rids), "hidden": torch.stack(hidden), "boundary_pos": list(bpos)},
                       os.path.join(args.out, name))
            shards.append({"file": name, "n": len(rids)})
            rids.clear(); hidden.clear(); bpos.clear()

        for start in range(0, len(ds), args.batch_size):
            batch = collate([ds[i] for i in range(start, min(start + args.batch_size, len(ds)))])
            ids = batch["teacher_input_ids"].cuda()
            mask = batch["teacher_attention_mask"].cuda()
            pos = torch.arange(ids.size(1), device="cuda").unsqueeze(0).expand_as(ids)
            b = batch["teacher_boundary"].cuda()
            with torch.no_grad(), model.capture_layer_outputs_at(b, layer_ids) as (store, _):
                model.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            h = torch.stack([store[l] for l in layer_ids], dim=1).to(torch.bfloat16).cpu()  # [B, n_layers, d]
            for i, rid in enumerate(batch["record_ids"]):
                rids.append(rid); hidden.append(h[i]); bpos.append(int(b[i]))
                if len(rids) >= args.shard_size:
                    flush()
            if start % (args.batch_size * 200) == 0:
                print(f"{split} {start}/{len(ds)} {time.time() - t0:.0f}s", flush=True)
        flush()
        manifest["splits"][split] = {"source": os.path.abspath(path), "n": len(ds), "shards": shards,
                                     "seconds": round(time.time() - t0)}
    dump_json(manifest, os.path.join(args.out, "manifest.json"))
    print("CACHE_DONE", {k: v["n"] for k, v in manifest["splits"].items()})


if __name__ == "__main__":
    main()
