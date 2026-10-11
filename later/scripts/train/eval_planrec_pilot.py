"""Final evaluation of one model for the two-loss PlanRec pilot (forward + greedy generation; no training).

  * fixed first-N val records (identical tokenization/targets/masks for every condition): execution CE and
    PlanRec CE (all / content / first-32 content / math tokens) for learned, zero, mean, shuffled latents
    (PlanRecMonitor, same code as the training probes); paired bootstrap 95% CIs of CE(cond) - CE(learned),
    token-weighted and per-example; cross-question cosine of Z.
  * free generation Q -> Z -> E, A on a fixed contamination-checked set: learned / mean / shuffled / zero latents
    and the explicit Q -> P -> E path; accuracy, completion, length, latency. The PlanRec branch is never used.
  * diagnostics only: greedy plan text reconstructed from Z alone (learned / shuffled / mean) for a few records.
Model "step0" = Stage-1 checkpoint + the run's saved step-0 projector.
"""
import argparse
import json
import os
import re
import time

import pandas as pd
import torch

from later.src.eval.eval_stage2 import analyse, verify
from later.src.train.stage2_common import dump_json, load_stage2_model, load_tokenizer
from later.src.train.stage2_plan_dataset import PLAN_OPEN, SEPARATOR, render_prompt, stage2_tokens
from later.src.train.stage2_planrec_monitor import CONDITIONS, PlanRecMonitor, bootstrap_gain

KEYS = {"exec": "exec_ce", "exec_nonrep": "exec_nonrep_ce", "rec": "planrec_ce", "rec_content": "planrec_content_ce",
        "rec_content32": "planrec_content32_ce", "rec_math": "planrec_math_ce"}


def token_f1(a, b):
    ta, tb = re.findall(r"\w+|\S", a.lower()), re.findall(r"\w+|\S", b.lower())
    if not ta or not tb:
        return 0.0
    from collections import Counter
    ov = sum((Counter(ta) & Counter(tb)).values())
    if ov == 0:
        return 0.0
    p, r = ov / len(ta), ov / len(tb)
    return 2 * p * r / (p + r)


def math_recall(gold, pred):
    spans = set(s.strip() for s in re.findall(r"\$([^$]+)\$", gold) if len(s.strip()) > 1)
    return (sum(s in pred for s in spans) / len(spans)) if spans else None


@torch.no_grad()
def generation(model, tok, problems, L, Zmean, max_new, bs):
    T = stage2_tokens(tok)
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    res, rows = {}, []
    for cond in ("learned", "mean", "shuffled", "zero", "explicit"):
        t0, n_tok = time.time(), 0
        for s in range(0, len(problems), bs):
            chunk = problems[s: s + bs]
            explicit = cond == "explicit"
            prompts = [tok.encode(render_prompt(tok, p["question"]), add_special_tokens=False) + ([] if explicit else T.plan_open_ids)
                       for p in chunk]
            kw = dict(latent_mode=cond if cond in ("learned", "shuffled", "zero") else "learned")
            if cond == "mean":
                kw["latent_vectors"] = Zmean.unsqueeze(0).expand(len(chunk), -1, -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                g = model.generate_plan_latent(prompts, 0 if explicit else L, [] if explicit else T.suffix_ids, max_new,
                                               eos, tok.pad_token_id, **(kw if not explicit else {}))
            for p, ids, stopped in zip(chunk, g["token_ids"], g["stopped"]):
                text = tok.decode(ids, skip_special_tokens=False)
                a = analyse(text, explicit, stopped)
                rows.append({"cond": cond, "id": p.get("problem_id", ""), "correct": bool(verify(a["pred"], p["answer"])),
                             "complete": bool(a["complete"]), "len": len(ids), "pred": a["pred"], "gold": p["answer"],
                             "text_head": text[:600]})
                n_tok += len(ids)
        dt = time.time() - t0
        rs = [r for r in rows if r["cond"] == cond]
        res[cond] = {"acc": sum(r["correct"] for r in rs) / len(rs), "complete": sum(r["complete"] for r in rs) / len(rs),
                     "mean_len": sum(r["len"] for r in rs) / len(rs), "seconds": dt, "seconds_per_problem": dt / len(rs),
                     "tokens_per_second": n_tok / dt}
        print("GEN", cond, json.dumps(res[cond]), flush=True)
    return res, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True, help="checkpoint dir, or the Stage-1 dir for step0")
    ap.add_argument("--projector", default=None, help="projector_step0.pt (step0 model only)")
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--gen_data", required=True)
    ap.add_argument("--gen_n", type=int, default=50)
    ap.add_argument("--gen_max_new_tokens", type=int, default=3072)
    ap.add_argument("--gen_batch_size", type=int, default=25)
    ap.add_argument("--n_rec_samples", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = load_tokenizer(args.stage1)
    model = load_stage2_model(args.model, latent_embed_builder_no_grad=False, latent_projector_dropout=0.0,
                              dtype=torch.float32, latent_output_rms="token_embedding").cuda().eval()
    if args.projector:
        model.latent_projector.load_state_dict(torch.load(args.projector, map_location="cuda"))
    L = 8
    val = pd.read_parquet(args.val)
    mon = PlanRecMonitor(model, tok, val, L, probe_n=args.n, probe_batch_size=8, grad_batch_size=4)
    t0 = time.time()
    res = mon.probe(return_records=True)
    recs = res.pop("_records")
    out = {"name": args.name, "model": args.model, "projector": args.projector, "n_records": len(recs),
           "probe_record_ids": mon.probe_ids, "metrics": res, "probe_seconds": time.time() - t0, "ci": {}}
    for key in KEYS:
        for cond in ("shuffled", "mean", "zero"):
            out["ci"][f"{KEYS[key]}_gain_{cond}"] = bootstrap_gain(recs, key, cond)
            out["ci"][f"{KEYS[key]}_gain_{cond}_per_example"] = bootstrap_gain(recs, key, cond, per_example=True)
    out["grad_split"] = mon.grad_split()
    Z = mon.last_Z                                            # [N, L, d] learned latents of the val records
    # ---- plan reconstruction samples (diagnostic only) ----
    T = stage2_tokens(tok)
    stop = [tok.encode("</", add_special_tokens=False)[0], tok.convert_tokens_to_ids("<|im_end|>")]
    samples = []
    idx = list(range(min(args.n_rec_samples, len(mon.rows))))
    from later.src.train.stage2_planrec_monitor import fixed_derangement
    perm = mon.perm
    variants = {"learned": Z[idx], "shuffled": Z[[perm[i] for i in idx]], "mean": Z.mean(0, keepdim=True).expand(len(idx), -1, -1)}
    gens = {}
    for v, z in variants.items():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ids = model.generate_plan_reconstruction(z.cuda(), T.plan_open_ids, 256, stop)
        gens[v] = [tok.decode(x, skip_special_tokens=True).split("</")[0] for x in ids]
    for k, i in enumerate(idx):
        r = mon.rows[i]
        s = {"record_id": r["record_id"], "question": r["question"][:500], "gold_plan": r["plan"],
             "shuffled_source_record": mon.rows[perm[i]]["record_id"]}
        for v in variants:
            s[f"rec_{v}"] = gens[v][k]
            s[f"f1_{v}"] = token_f1(gens[v][k], r["plan"])
            s[f"math_recall_{v}"] = math_recall(r["plan"], gens[v][k])
        s["f1_shuffled_vs_its_source_plan"] = token_f1(gens["shuffled"][k], mon.rows[perm[i]]["plan"])
        samples.append(s)
    out["plan_reconstruction_summary"] = {
        f"mean_f1_{v}": sum(s[f"f1_{v}"] for s in samples) / len(samples) for v in variants}
    for v in variants:
        mr = [s[f"math_recall_{v}"] for s in samples if s[f"math_recall_{v}"] is not None]
        out["plan_reconstruction_summary"][f"mean_math_recall_{v}"] = sum(mr) / max(len(mr), 1)
    # ---- free generation ----
    gres, grows = {}, []
    if args.gen_n > 0:
        probs = pd.read_parquet(args.gen_data).iloc[: args.gen_n].to_dict("records")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Zg = model._latents_for_prompts([tok.encode(render_prompt(tok, p["question"]), add_special_tokens=False) + T.plan_open_ids
                                             for p in probs], L, tok.pad_token_id).float()
        gres, grows = generation(model, tok, probs, L, Zg.mean(0), args.gen_max_new_tokens, args.gen_batch_size)
    out["generation"] = gres
    out["generation_note"] = ("mean = per-slot mean of the learned Z over the gen problems; shuffled = each problem gets "
                              "the next problem's Z within its decoding batch (batch size %d)" % args.gen_batch_size)
    os.makedirs(args.out, exist_ok=True)
    dump_json(out, os.path.join(args.out, f"metrics_{args.name}.json"))
    with open(os.path.join(args.out, f"samples_{args.name}.jsonl"), "w") as f:
        for s in samples:
            f.write(json.dumps({"model": args.name, "kind": "plan_reconstruction", **s}, ensure_ascii=False) + "\n")
        for r in grows:
            f.write(json.dumps({"model": args.name, "kind": "generation", **r}, ensure_ascii=False) + "\n")
    rows = []
    for rid, rc in recs.items():
        for cond in CONDITIONS:
            d = {"model": args.name, "record_id": rid, "condition": cond}
            for key, name in KEYS.items():
                n = rc[cond][f"{key}_n"]
                d[name] = rc[cond][f"{key}_sum"] / n if n else None
                d[f"{name}_tokens"] = n
            rows.append(d)
    pd.DataFrame(rows).to_csv(os.path.join(args.out, f"per_example_{args.name}.csv"), index=False)
    print("EVAL_DONE", args.name, flush=True)


if __name__ == "__main__":
    main()
