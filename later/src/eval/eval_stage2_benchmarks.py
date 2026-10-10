"""Full benchmark evaluation for OUR Stage 2 (greedy, same decoder for every condition; resumable).

Conditions (same parser/verifier as eval_stage2.py):
  stage1_explicit     Stage-1 SFT  Q -> P -> E -> A
  stage1_empty_plan   Stage-1 SFT  Q -> '<EXPLORATION>\\n' + separator -> E -> A             (aux)
  stage2_latent       Stage-2      Q -> z_1..z_L -> separator -> E -> A                       (main)
  stage2_explicit     Stage-2      Q -> P -> E -> A                                          (anti-forgetting)
  stage2_zero_latent  Stage-2      same L positions, zero latent embeddings
  stage2_shuffled     Stage-2      same L positions, latents of another problem in the batch
  stage2_no_latent    Stage-2      L = 0, separator right after '<EXPLORATION>\\n'              (aux)
Each problem is flagged `contaminated` if it has a normalized exact match or a shared 13-word n-gram with any row of
the E2C SFT source (the Stage-1 checkpoint was trained on that data); metrics are reported for all and for clean.
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
from later.src.train.stage2_plan_dataset import render_prompt, stage2_tokens

CONDS = {  # name: (model, latent_steps?, suffix?, plan_open?, latent_mode, explicit output)
    "stage1_explicit": ("stage1", False, False, False, "learned", True),
    "stage1_empty_plan": ("stage1", False, True, True, "learned", False),
    "stage2_latent": ("stage2", True, True, True, "learned", False),
    "stage2_explicit": ("stage2", False, False, False, "learned", True),
    "stage2_zero_latent": ("stage2", True, True, True, "zero", False),
    "stage2_shuffled": ("stage2", True, True, True, "shuffled", False),
    "stage2_no_latent": ("stage2", False, True, True, "learned", False),
}


def norm_q(q):
    return re.sub(r"\s+", " ", str(q).strip().lower())


def ngrams(s, n=13):
    w = re.findall(r"\w+", str(s).lower())
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def summarize(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    r = lambda k: sum(bool(x[k]) for x in rows) / n
    return {"n": n, "accuracy": r("correct"), "complete_cot_rate": r("complete"), "exec_closed_rate": r("exec_closed"),
            "answer_parse_rate": sum(x["pred"] is not None for x in rows) / n, "eos_stop_rate": r("stopped_with_eos"),
            "mean_gen_tokens": sum(x["gen_tokens"] for x in rows) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--sft_source", required=True)
    ap.add_argument("--datasets", required=True, help="comma list, e.g. aime24,aime25")
    ap.add_argument("--conditions", default=",".join(CONDS))
    ap.add_argument("--aux_conditions_skip", default="", help="dataset:cond,... pairs to skip")
    ap.add_argument("--latent_steps", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=3072)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok = load_tokenizer(args.stage1)
    T = stage2_tokens(tok)
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    skip = {tuple(x.split(":")) for x in args.aux_conditions_skip.split(",") if x}

    sft = pd.read_parquet(args.sft_source)
    sft_q = set(sft.question.map(norm_q))
    sft_ng = set()
    for q in sft.question:
        sft_ng |= ngrams(q)

    models = {}
    for ds in args.datasets.split(","):
        frame = pd.read_parquet(os.path.join(args.eval_dir, f"{ds}.parquet"))
        if args.limit:
            frame = frame.iloc[: args.limit]
        probs = [{"id": f"{ds}-{i}", "question": q, "answer": str(a),
                  "contaminated": bool(norm_q(q) in sft_q or (ngrams(q) & sft_ng))}
                 for i, (q, a) in enumerate(zip(frame.question, frame.answer))]
        order = sorted(range(len(probs)), key=lambda i: len(probs[i]["question"]))
        for cond in args.conditions.split(","):
            if (ds, cond) in skip:
                continue
            path = os.path.join(args.out, f"samples_{ds}_{cond}.jsonl")
            if os.path.exists(path) and sum(1 for _ in open(path)) == len(probs):
                print(f"skip {ds} {cond} (done)", flush=True)
                continue
            which, use_L, use_sfx, add_open, mode, explicit = CONDS[cond]
            if which not in models:
                models.clear()
                torch.cuda.empty_cache()
                models[which] = load_stage2_model(args.stage1 if which == "stage1" else args.stage2,
                                                  latent_embed_builder_no_grad=True, latent_projector_dropout=0.0,
                                                  dtype=torch.bfloat16).cuda().eval()
            m = models[which]
            steps = args.latent_steps if use_L else 0
            sfx = T.suffix_ids if use_sfx else []
            chunks = [order[s: s + args.batch_size] for s in range(0, len(order), args.batch_size)]
            if mode == "shuffled" and len(chunks) > 1 and len(chunks[-1]) < 2:   # shuffled needs >= 2 per batch
                tail = chunks.pop()
                chunks[-1] = chunks[-1] + tail
            rows, t0 = [], time.time()
            for ch in chunks:
                prompts = [tok.encode(render_prompt(tok, probs[i]["question"]), add_special_tokens=False)
                           + (T.plan_open_ids if add_open else []) for i in ch]
                g = m.generate_plan_latent(prompts, steps, sfx, args.max_new_tokens, eos, tok.pad_token_id,
                                           latent_mode=mode)
                for i, ids, stopped in zip(ch, g["token_ids"], g["stopped"]):
                    text = tok.decode(ids, skip_special_tokens=False)
                    a = analyse(text, explicit, stopped)
                    a["correct"] = verify(a["pred"], probs[i]["answer"])
                    rows.append({**probs[i], "condition": cond, "output": text, "gen_tokens": len(ids),
                                 "stopped_with_eos": stopped, **a})
            rows.sort(key=lambda x: int(x["id"].rsplit("-", 1)[1]))
            with open(path, "w") as f:
                for x in rows:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
            res = {"all": summarize(rows), "clean": summarize([x for x in rows if not x["contaminated"]]),
                   "seconds": round(time.time() - t0)}
            dump_json(res, os.path.join(args.out, f"metrics_{ds}_{cond}.json"))
            print(ds, cond, json.dumps({"all_acc": res["all"]["accuracy"], "clean_acc": res["clean"].get("accuracy"),
                                        "n": res["all"]["n"], "n_clean": res["clean"]["n"],
                                        "complete": res["all"]["complete_cot_rate"],
                                        "len": round(res["all"]["mean_gen_tokens"]), "sec": res["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
