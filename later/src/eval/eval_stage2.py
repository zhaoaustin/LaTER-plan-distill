"""P4 evaluation for OUR Stage 2 on the same held-out problems, same greedy decoder for every condition.

Conditions
  stage1_explicit     Stage-1 SFT, Q -> P_text -> E -> A            (reference)
  stage1_empty_plan   Stage-1 SFT, Q -> '<EXPLORATION>\\n' + separator -> E -> A   (no-plan floor)
  stage2_latent       Stage-2, Q -> '<EXPLORATION>\\n' z_1..z_L separator -> E -> A   (main)
  stage2_explicit     Stage-2, Q -> P_text -> E -> A               (anti-forgetting check)
  stage2_zero_latent  Stage-2, same L positions but all-zero latent embeddings (is the latent used?)
  stage2_no_latent    Stage-2, L=0, separator forced right after '<EXPLORATION>\\n'
Problem sets
  val     : held-out Stage-2 val records; reference = the Stage-1 target answer (agreement, not gold accuracy)
  math500 : MATH-500 with gold answers (train records overlapping any eval set were removed in the audit)
"""
import argparse
import json
import os
import random
import time

import pandas as pd
import torch

from later.src.train.stage2_common import dump_json, load_stage2_model, load_tokenizer
from later.src.train.stage2_plan_dataset import render_prompt, stage2_tokens

XB, XE, EB, EE = "<EXECUTION>", "</EXECUTION>", "<EXPLORATION>", "</EXPLORATION>"


def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    depth = 0
    for k in range(j, len(s)):
        depth += s[k] == "{"
        depth -= s[k] == "}"
        if depth == 0:
            return s[j + 1:k]
    return None


def verify(pred, gold):
    if pred is None:
        return False
    try:
        from math_verify import parse, verify as mv
        return bool(mv(parse("$" + gold + "$"), parse("$" + pred + "$")))
    except Exception:
        return pred.replace(" ", "") == gold.replace(" ", "")


def analyse(text, explicit, stopped):
    t = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    if explicit:
        has_plan = t.startswith(EB) and EE in t and XB in t and t.find(EE) < t.find(XB)
        exe = t.split(XB, 1)[1] if XB in t else ""
    else:
        has_plan = None
        exe = t
    closed = XE in exe
    exe = exe.split(XE, 1)[0]
    pred = last_boxed(exe)
    derivation = len(exe.strip()) > 200 and any(c in exe for c in "=+-*/")
    return {"has_plan": has_plan, "exec_closed": closed, "pred": pred, "has_derivation": derivation,
            "new_plan_after_latent": (EB in t or EE in t) if not explicit else None,
            "complete": bool(closed and pred is not None and derivation and stopped and (has_plan is not False))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--math500", required=True)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--n_math", type=int, default=100)
    ap.add_argument("--latent_steps", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=3072)
    ap.add_argument("--batch_size", type=int, default=25)
    ap.add_argument("--conditions", default="all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok = load_tokenizer(args.stage1)
    T = stage2_tokens(tok)
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    val = pd.read_parquet(args.val)
    vi = sorted(random.Random(0).sample(range(len(val)), min(args.n_val, len(val))))
    m5 = pd.read_parquet(args.math500)
    mi = list(range(len(m5)))
    random.Random(0).shuffle(mi)
    sets = {"val": [(val.record_id.iloc[i], val.question.iloc[i], val.answer.iloc[i]) for i in vi],
            "math500": [(f"math500-{i}", m5.question.iloc[i], m5.answer.iloc[i]) for i in mi[: args.n_math]]}
    L = args.latent_steps
    conds = {  # name: (which model, latent_steps, suffix, add plan_open, latent_mode, explicit output)
        "stage1_explicit": ("stage1", 0, [], False, "learned", True),
        "stage1_empty_plan": ("stage1", 0, T.suffix_ids, True, "learned", False),
        "stage2_latent": ("stage2", L, T.suffix_ids, True, "learned", False),
        "stage2_explicit": ("stage2", 0, [], False, "learned", True),
        "stage2_zero_latent": ("stage2", L, T.suffix_ids, True, "zero", False),
        "stage2_no_latent": ("stage2", 0, T.suffix_ids, True, "learned", False),
    }
    if args.conditions != "all":
        conds = {k: v for k, v in conds.items() if k in args.conditions.split(",")}
    metrics = {"latent_steps": L, "max_new_tokens": args.max_new_tokens, "decoding": "greedy",
               "stage1": args.stage1, "stage2": args.stage2, "n": {k: len(v) for k, v in sets.items()}, "results": {}}
    models = {}
    for name, (which, steps, sfx, add_open, mode, explicit) in conds.items():
        if which not in models:
            models.clear()
            torch.cuda.empty_cache()
            models[which] = load_stage2_model(args.stage1 if which == "stage1" else args.stage2,
                                              latent_embed_builder_no_grad=True, latent_projector_dropout=0.0,
                                              dtype=torch.bfloat16).cuda().eval()
        m = models[which]
        for set_name, items in sets.items():
            t0 = time.time()
            rows = []
            for s in range(0, len(items), args.batch_size):
                chunk = items[s: s + args.batch_size]
                prompts = [tok.encode(render_prompt(tok, q), add_special_tokens=False) + (T.plan_open_ids if add_open else [])
                           for _, q, _ in chunk]
                g = m.generate_plan_latent(prompts, steps, sfx, args.max_new_tokens, eos, tok.pad_token_id,
                                           latent_mode=mode)
                for (rid, q, ref), ids, stopped in zip(chunk, g["token_ids"], g["stopped"]):
                    text = tok.decode(ids, skip_special_tokens=False)
                    a = analyse(text, explicit, stopped)
                    a["correct"] = verify(a["pred"], ref)
                    rows.append({"id": rid, "question": q, "reference": ref, "output": text, "gen_tokens": len(ids),
                                 "stopped_with_eos": stopped, **a})
            n = len(rows)
            r = lambda k: sum(bool(x[k]) for x in rows) / n
            res = {"accuracy_or_agreement": r("correct"), "complete_cot_rate": r("complete"),
                   "exec_closed_rate": r("exec_closed"), "answer_parse_rate": sum(x["pred"] is not None for x in rows) / n,
                   "eos_stop_rate": r("stopped_with_eos"), "mean_gen_tokens": sum(x["gen_tokens"] for x in rows) / n,
                   "seconds": round(time.time() - t0)}
            if explicit:
                res["plan_present_rate"] = r("has_plan")
            else:
                res["regenerated_plan_tags_rate"] = r("new_plan_after_latent")
            metrics["results"].setdefault(name, {})[set_name] = res
            with open(os.path.join(args.out, f"samples_{name}_{set_name}.jsonl"), "w") as f:
                for x in rows:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
            print(name, set_name, json.dumps(res), flush=True)
            dump_json(metrics, os.path.join(args.out, "eval_metrics.json"))


if __name__ == "__main__":
    main()
