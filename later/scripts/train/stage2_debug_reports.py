"""P1/P2 deliverables for OUR Stage 2: batch_debug.json, gradient_report.json, 20 three-view samples (markdown).

Gradient report: per loss term in isolation (execution-only, alignment-only, explicit-only) and full, for
L in {2,4,8}, with and without gradient checkpointing. Teacher targets are computed online from θ0 (the
student is still θ0 here, so this equals the frozen-teacher cache).
"""
import argparse
import json
import os
import random

import pandas as pd
import torch

from later.src.train.stage2_common import dump_json, load_stage2_model, load_tokenizer
from later.src.train.stage2_objective import Stage2Module
from later.src.train.stage2_plan_dataset import Stage2Collator, build_views, stage2_tokens


def gnorm(params):
    gs = [p.grad.float() for p in params if p.grad is not None]
    return float(torch.sqrt(sum((g ** 2).sum() for g in gs))) if gs else 0.0


def online_teacher(model, batch, layers):
    ids = batch["teacher_input_ids"].cuda()
    with torch.no_grad(), model.capture_layer_outputs_at(batch["teacher_boundary"].cuda(), layers) as (s, _):
        model.model(input_ids=ids, attention_mask=batch["teacher_attention_mask"].cuda(),
                    position_ids=torch.arange(ids.size(1), device="cuda").unsqueeze(0).expand_as(ids), use_cache=False)
    return {l: s[l].clone() for l in layers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok = load_tokenizer(args.ckpt)
    T = stage2_tokens(tok)
    frame = pd.read_parquet(os.path.join(args.data, "stage2_train.parquet"))

    # ---- 20 three-view samples for manual inspection
    rng = random.Random(0)
    idxs = sorted(rng.sample(range(len(frame)), 20))
    lines = ["# Stage 2 three-view samples (20 random train records, L=8)\n",
             "Segments: `-` not supervised, `P` plan, `E` execution, `A` answer (from last \\boxed), `C` closing tag/EOS, "
             "`T` plan-side tags. Latent slots shown as `<z>`.\n"]
    for i in idxs:
        r = frame.iloc[i].to_dict()
        v = build_views(tok, r, 8, T, 4096)
        def show(ids, labels=None, latent=()):
            parts = []
            for j, t in enumerate(ids):
                parts.append("<z>" if j in latent else tok.decode([t]))
            return "".join(parts)
        sup_s = [j for j, l in enumerate(v["student_labels"]) if l != -100]
        sup_x = [j for j, l in enumerate(v["explicit_labels"]) if l != -100]
        lines += [f"\n## {r['record_id']}  (source row {r['source_row']})\n",
                  f"**Q**\n```\n{r['question']}\n```\n**P**\n```\n{r['plan']}\n```\n**E** (first 600 chars)\n```\n{r['execution'][:600]}\n```\n**A** `{r['answer']}`\n",
                  f"**teacher_prefix** ({len(v['teacher_ids'])} tok, boundary b_T={v['teacher_boundary']} "
                  f"token={tok.convert_ids_to_tokens(v['teacher_ids'][-1])!r}) tail:\n```\n{tok.decode(v['teacher_ids'][-40:])}\n```\n",
                  f"**student_latent** ({len(v['student_ids'])} tok, latent_start={v['latent_start']}, slots={v['latent_positions'][0]}..{v['latent_positions'][-1]}, "
                  f"boundary b_S={v['student_boundary']}); supervised tokens {sup_s[0]}..{sup_s[-1]} ({len(sup_s)}): "
                  f"E={v['student_seg'].count('E')} A={v['student_seg'].count('A')} C={v['student_seg'].count('C')} P={v['student_seg'].count('P')}\n"
                  f"```\n{show(v['student_ids'][v['prompt_len']-5: v['student_boundary'] + 25], latent=[p - (v['prompt_len'] - 5) for p in v['latent_positions']])}\n```\n",
                  f"first supervised student target = {tok.decode([v['student_labels'][v['student_boundary'] + 1]])!r} "
                  f"(predicted from boundary position {v['student_boundary']})\n",
                  f"**student_explicit** ({len(v['explicit_ids'])} tok); supervised {sup_x[0]}..{sup_x[-1]} ({len(sup_x)}): "
                  f"P={v['explicit_seg'].count('P')} E={v['explicit_seg'].count('E')} A={v['explicit_seg'].count('A')} "
                  f"C={v['explicit_seg'].count('C')} T={v['explicit_seg'].count('T')}\n"]
    open(os.path.join(args.out, "stage2_three_view_samples.md"), "w").write("".join(lines))

    # ---- batch_debug.json (one real collated batch of 2)
    views = [build_views(tok, frame.iloc[i].to_dict(), 8, T, 4096) for i in idxs[:2]]
    batch = Stage2Collator(tok.pad_token_id)(views)
    dbg = {"record_ids": batch["record_ids"]}
    for k, val in batch.items():
        if torch.is_tensor(val):
            dbg[k] = {"shape": list(val.shape), "dtype": str(val.dtype)}
    for b in range(2):
        sl = batch["student_labels"][b]
        first = int((sl != -100).nonzero()[0])
        dbg[f"row{b}"] = {
            "student_boundary": int(batch["student_boundary"][b]), "first_supervised_index": first,
            "boundary_token": tok.convert_ids_to_tokens(int(batch["student_input_ids"][b, first - 1])),
            "first_target": tok.decode([int(sl[first])]),
            "latent_positions": batch["latent_positions"][b].tolist(),
            "latent_start_token": tok.convert_ids_to_tokens(int(batch["student_input_ids"][b, batch["latent_start_positions"][b]])),
            "n_student_supervised": int((sl != -100).sum()), "n_explicit_supervised": int((batch["explicit_labels"][b] != -100).sum()),
            "teacher_len": int(batch["teacher_attention_mask"][b].sum()), "teacher_boundary": int(batch["teacher_boundary"][b]),
            "student_len": int(batch["student_attention_mask"][b].sum()), "explicit_len": int(batch["explicit_attention_mask"][b].sum()),
            "rope_position_at_boundary": {"teacher": int(batch["teacher_boundary"][b]),
                                          "student": int(batch["student_position_ids"][b, batch["student_boundary"][b]])},
        }
    dump_json(dbg, os.path.join(args.out, "batch_debug.json"))

    # ---- gradient report
    model = load_stage2_model(args.ckpt, latent_embed_builder_no_grad=False, latent_projector_dropout=0.0,
                              dtype=torch.bfloat16).cuda()
    model.train()
    layers = list(range(1, model.config.num_hidden_layers + 1))
    proj = list(model.latent_projector.parameters())
    back = [p for n, p in model.named_parameters() if n.startswith("model.layers.")]
    emb = [model.get_input_embeddings().weight]
    report = {"model_dtype": "bfloat16", "latent_embed_builder_no_grad": model.config.latent_embed_builder_no_grad,
              "alignment_layers": "all (1..36, decoder outputs pre final norm)", "records": batch["record_ids"], "runs": []}
    configs = {"execution_only": dict(lambda_a=0.0, lambda_r=0.0, execution_ce_weight=1.0),
               "alignment_only": dict(lambda_a=1.0, lambda_r=0.0, execution_ce_weight=0.0),
               "explicit_only": dict(lambda_a=0.0, lambda_r=1.0, execution_ce_weight=0.0),
               "full_smoke_weights": dict(lambda_a=0.1, lambda_r=0.3, execution_ce_weight=1.0)}
    for gc in (False, True):
        if gc:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for L in (2, 4, 8):
            views = [build_views(tok, frame.iloc[i].to_dict(), L, T, 4096) for i in idxs[:2]]
            batch = Stage2Collator(tok.pad_token_id)(views)
            teacher = online_teacher(model, batch, layers)
            for name, c in configs.items():
                model.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                mod = Stage2Module(model, dict(c, alignment_epsilon=1e-6), layers)
                out = mod(batch, teacher, debug_latent_grad=True)
                out["loss"].backward()
                po = out.get("latent", {}).get("projector_outputs", [])
                z_grads = [float(z.grad.float().norm()) if z.grad is not None else 0.0 for z in po]
                report["runs"].append({
                    "gradient_checkpointing": gc, "latent_steps": L, "loss_config": name,
                    "L_execution": float(out["l_exec"]), "L_alignment": float(out["l_align"]),
                    "L_explicit": float(out["l_explicit"]), "L_total": float(out["loss"]),
                    "grad_norm_latent_projector": gnorm(proj), "grad_norm_backbone_layers": gnorm(back),
                    "grad_norm_embedding": gnorm(emb), "latent_step_grad_norms": z_grads,
                    "early_latent_grad_norm": z_grads[0] if z_grads else 0.0,
                    "all_grads_finite": all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None),
                    "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2 ** 30,
                })
                print(json.dumps({k: report["runs"][-1][k] for k in ("gradient_checkpointing", "latent_steps", "loss_config",
                                  "grad_norm_latent_projector", "early_latent_grad_norm", "grad_norm_backbone_layers")}))
        if gc:
            model.gradient_checkpointing_disable()
    model.zero_grad(set_to_none=True)
    dump_json(report, os.path.join(args.out, "gradient_report.json"))


if __name__ == "__main__":
    main()
