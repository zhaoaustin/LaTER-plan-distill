"""Frozen Stage-1 teacher execution-token distributions for the Stage-2 execution-KL pilot.

Teacher input = the Stage-1 explicit sequence  Q | <EXPLORATION>\\n P \\n</EXPLORATION><EXECUTION>\\n E ...
For every KL target (execution offset j from build_views.kl_offsets) the teacher's next-token distribution is read
at position b_X + j, i.e. it has seen Q, P, the boundary and E_<j only (causal). Stored: top-k token ids + probs
and the explicit tail mass 1 - sum(top-k) (temperature 1.0 by default). Independent of the boundary-hidden cache.

Records: either --record_ids_json (training records of the pilot steps) or --first N of a split (eval probe).
"""
import argparse
import json
import os
import time

import pandas as pd
import torch

from later.src.train.stage2_common import checkpoint_hash, dump_json, load_stage2_model, load_tokenizer, tokenizer_hashes
from later.src.train.stage2_plan_dataset import BOUNDARY_RULE_VERSION, KL_MASK_RULE, build_views, stage2_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split_name", required=True, help="train_pilot | val_probe")
    ap.add_argument("--record_ids_json", default=None)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--kl_budget", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok = load_tokenizer(args.teacher)
    T = stage2_tokens(tok)
    frame = pd.read_parquet(args.data)
    if args.record_ids_json:
        want = json.load(open(args.record_ids_json))
        pos = {r: i for i, r in enumerate(frame.record_id)}
        rows = [frame.iloc[pos[r]].to_dict() for r in dict.fromkeys(want)]   # unique, keep order
    else:
        rows = [frame.iloc[i].to_dict() for i in range(args.first)]
    model = load_stage2_model(args.teacher, latent_embed_builder_no_grad=True, latent_projector_dropout=0.0,
                              dtype=torch.bfloat16).cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    out, t0, cov, n_pos, zero = {}, time.time(), [], 0, 0
    with torch.no_grad():
        for s in range(0, len(rows), args.batch_size):
            views = [build_views(tok, r, 1, T, 4096, kl_budget=args.kl_budget) for r in rows[s: s + args.batch_size]]
            n = max(len(v["explicit_ids"]) for v in views)
            ids = torch.full((len(views), n), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(views), n), dtype=torch.long)
            for i, v in enumerate(views):
                ids[i, : len(v["explicit_ids"])] = torch.tensor(v["explicit_ids"])
                mask[i, : len(v["explicit_ids"])] = 1
            pos_ids = torch.arange(n).unsqueeze(0).expand_as(ids)
            hid = model.model(input_ids=ids.cuda(), attention_mask=mask.cuda(), position_ids=pos_ids.cuda(),
                              use_cache=False).last_hidden_state
            for i, v in enumerate(views):
                offs = v["kl_offsets"]
                if not offs:
                    zero += 1
                    out[v["record_id"]] = {"offsets": torch.zeros(0, dtype=torch.int16)}
                    continue
                src = torch.tensor([v["explicit_boundary"] + j for j in offs], device="cuda")
                logits = model.lm_head(hid[i, src]).float() / args.temperature
                probs = torch.softmax(logits, dim=-1)
                tp, ti = probs.topk(args.k, dim=-1)
                tail = (1.0 - tp.sum(-1)).clamp(min=0.0)
                target = torch.tensor([v["explicit_ids"][v["explicit_boundary"] + 1 + j] for j in offs])
                out[v["record_id"]] = {"offsets": torch.tensor(offs, dtype=torch.int16), "ids": ti.int().cpu(),
                                       "probs": tp.cpu(), "tail": tail.cpu(), "target": target.int()}
                cov.append(float(tp.sum(-1).mean()))
                n_pos += len(offs)
            if s % (args.batch_size * 50) == 0:
                print(f"{args.split_name} {s}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    torch.save(out, os.path.join(args.out, f"{args.split_name}.pt"))
    man_path = os.path.join(args.out, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}
    man.update({"teacher_checkpoint": os.path.abspath(args.teacher), "teacher_checkpoint_hash": checkpoint_hash(args.teacher),
                **tokenizer_hashes(tok, args.teacher), "boundary_rule_version": BOUNDARY_RULE_VERSION,
                "kl_mask_rule": KL_MASK_RULE, "kl_budget": args.kl_budget, "k": args.k, "temperature": args.temperature,
                "kl_direction": "forward KL(teacher || student)"})
    man.setdefault("splits", {})[args.split_name] = {
        "source": os.path.abspath(args.data), "n_records": len(rows), "n_positions": n_pos,
        "records_without_kl_targets": zero, "mean_topk_mass": sum(cov) / max(1, len(cov)),
        "seconds": round(time.time() - t0), "record_ids": [r["record_id"] for r in rows]}
    dump_json(man, man_path)
    print("EXEC_KL_CACHE_DONE", args.split_name, len(rows), n_pos, f"topk_mass={sum(cov) / max(1, len(cov)):.4f}",
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
