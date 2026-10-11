"""Test 1 (diagnostics): is the explicit-mode 0/50 of pilot A/B forgetting or an evaluation-interface artefact?

One run = (model checkpoint, loading path, protocol) on the same fixed problems and prompts:
  load=native   : transformers Qwen3ForCausalLM.from_pretrained (latent projector weights ignored), bf16 weights
  load=wrapper  : later LatentQwenForCausalLM via load_stage2_model, fp32 weights + bf16 autocast (pilot eval path)
  protocol=native_sft : HF model.generate, greedy, left padding, batch 25 (Stage-1 eval_e2c.py protocol)
  protocol=stage2     : model.generate_plan_latent(latent_steps=0, suffix=[]) greedy loop (pilot eval path)
Prompt = Qwen3 chat template (user question, add_generation_prompt=True), nothing appended; max_new_tokens and EOS
(<|im_end|>, <|endoftext|>) identical. Every output is parsed three ways: the Stage-1 parser (tags anywhere), the
pilot parser (must start with <EXPLORATION>), and format-free math (last \\boxed{} anywhere in the output).
--compare_logits: next-token distributions at the end of every prompt, native vs wrapper, plus first 10 greedy tokens.
"""
import argparse
import importlib.util
import json
import os
import time

import pandas as pd
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from later.src.eval.eval_stage2 import analyse, last_boxed, verify
from later.src.train.stage2_common import load_stage2_model
from later.src.train.stage2_plan_dataset import render_prompt

EB, EE, XB, XE = "<EXPLORATION>", "</EXPLORATION>", "<EXECUTION>", "</EXECUTION>"
_spec = importlib.util.spec_from_file_location("eval_e2c", "/scratch/hz33227/e2c/Explore-Execute-Chain/tools_e2c4b/eval_e2c.py")
eval_e2c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_e2c)


def load(path, how):
    if how == "native":
        cfg = Qwen3Config.from_pretrained(path)
        return Qwen3ForCausalLM.from_pretrained(path, config=cfg, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    return load_stage2_model(path, latent_embed_builder_no_grad=True, latent_projector_dropout=0.0, dtype=torch.float32,
                             latent_output_rms="token_embedding").cuda().eval()


def classify(t, stopped):
    if not stopped:
        return "truncated_no_eos"
    if not t.lstrip().startswith(EB):
        return "missing_exploration_open" if EB not in t else "exploration_not_at_start"
    if EE not in t:
        return "missing_exploration_close"
    if XB not in t:
        return "missing_execution_open"
    if XE not in t:
        return "missing_execution_close"
    return "format_ok"


@torch.inference_mode()
def generate(model, tok, prompts, protocol, max_new, bs, eos):
    out = []
    for s in range(0, len(prompts), bs):
        chunk = prompts[s: s + bs]
        if protocol == "native_sft":
            tok.padding_side = "left"
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=eos, pad_token_id=tok.pad_token_id,
                               temperature=None, top_p=None, top_k=None)
            for k in range(len(chunk)):
                ids = g[k, enc.input_ids.shape[1]:].tolist()
                stopped = any(t in eos for t in ids)
                if stopped:
                    ids = ids[: min(ids.index(t) for t in eos if t in ids) + 1]
                out.append((ids, stopped))
        else:
            ids = [tok.encode(p, add_special_tokens=False) for p in chunk]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                g = model.generate_plan_latent(ids, 0, [], max_new, eos, tok.pad_token_id)
            out += list(zip(g["token_ids"], g["stopped"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--load", choices=["native", "wrapper"], required=True)
    ap.add_argument("--protocol", choices=["native_sft", "stage2"], required=True)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=3072)
    ap.add_argument("--bs", type=int, default=25)
    ap.add_argument("--compare_logits", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.stage1)
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    probs = pd.read_parquet(a.problems).iloc[: a.n].to_dict("records")
    prompts = [render_prompt(tok, p["question"]) for p in probs]
    if a.compare_logits:
        nat, wrp = load(a.model, "native"), load(a.model, "wrapper")
        res = []
        with torch.inference_mode():
            for p in prompts:
                ids = torch.tensor([tok.encode(p, add_special_tokens=False)], device="cuda")
                ln = nat(input_ids=ids).logits[0, -1].float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lw = wrp.model(input_ids=ids).last_hidden_state[0, -1]
                    lw = wrp.lm_head(lw).float()
                lwf = wrp.lm_head(wrp.model(input_ids=ids).last_hidden_state[0, -1]).float()   # fp32, no autocast
                pn, pw = torch.log_softmax(ln, -1), torch.log_softmax(lw, -1)
                res.append({"top1_native": tok.decode([int(ln.argmax())]), "top1_wrapper": tok.decode([int(lw.argmax())]),
                            "top1_wrapper_fp32": tok.decode([int(lwf.argmax())]),
                            "kl_native_wrapper": float((pn.exp() * (pn - pw)).sum()), "max_abs_logprob_diff_top100":
                            float((pn - pw)[pn.topk(100).indices].abs().max())})
            g_n = generate(nat, tok, prompts[:10], "native_sft", 10, 10, eos)
            g_w = generate(wrp, tok, prompts[:10], "stage2", 10, 10, eos)
        first10 = [{"native": tok.decode(x[0]), "wrapper_stage2": tok.decode(y[0]), "identical": x[0] == y[0]} for x, y in zip(g_n, g_w)]
        summ = {"n": len(res), "top1_agree": sum(r["top1_native"] == r["top1_wrapper"] for r in res) / len(res),
                "top1_agree_fp32": sum(r["top1_native"] == r["top1_wrapper_fp32"] for r in res) / len(res),
                "mean_kl": sum(r["kl_native_wrapper"] for r in res) / len(res), "max_kl": max(r["kl_native_wrapper"] for r in res),
                "first10_identical": sum(f["identical"] for f in first10), "top1_tokens": sorted({r["top1_native"] for r in res}),
                "per_prompt": res, "first10": first10}
        json.dump(summ, open(os.path.join(a.out, f"logits_{a.name}.json"), "w"), indent=1, ensure_ascii=False)
        print("LOGITS", a.name, json.dumps({k: v for k, v in summ.items() if k not in ("per_prompt", "first10")}, ensure_ascii=False))
        return
    model = load(a.model, a.load)
    t0 = time.time()
    gens = generate(model, tok, prompts, a.protocol, a.max_new_tokens, a.bs, eos)
    dt = time.time() - t0
    rows = []
    for p, prompt, (ids, stopped) in zip(probs, prompts, gens):
        text = tok.decode(ids, skip_special_tokens=False)
        t = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
        nat = eval_e2c.parse_output(t)
        pil = analyse(text, True, stopped)
        anyb = last_boxed(t)
        rows.append({"run": a.name, "problem_id": p["problem_id"], "gold": p["answer"], "gen_tokens": len(ids), "stopped": stopped,
                     "first10": tok.decode(ids[:10]), "has_exploration_open_at_start": t.lstrip().startswith(EB),
                     "has_exploration_open": EB in t, "has_exploration_close": EE in t, "has_execution_open": XB in t,
                     "has_execution_close": XE in t, "category": classify(t, stopped),
                     "native_format_valid": bool(nat["format_valid"]), "native_pred": nat["pred"], "native_correct": verify(nat["pred"], p["answer"]),
                     "pilot_complete": pil["complete"], "pilot_pred": pil["pred"], "pilot_correct": verify(pil["pred"], p["answer"]),
                     "anyboxed_pred": anyb, "anyboxed_correct": verify(anyb, p["answer"]),
                     "parsed_plan": nat["exploration"], "parsed_execution_head": (nat["execution"] or "")[:400], "raw_output": text})
    n = len(rows)
    r = lambda k: sum(bool(x[k]) for x in rows) / n
    m = {"run": a.name, "model": a.model, "load": a.load, "protocol": a.protocol, "n": n, "max_new_tokens": a.max_new_tokens,
         "seconds": dt, "eos_stop_rate": r("stopped"), "mean_gen_tokens": sum(x["gen_tokens"] for x in rows) / n,
         "native_format_valid_rate": r("native_format_valid"), "native_accuracy": r("native_correct"),
         "pilot_complete_rate": r("pilot_complete"), "pilot_accuracy": r("pilot_correct"),
         "anyboxed_parse_rate": sum(x["anyboxed_pred"] is not None for x in rows) / n, "anyboxed_accuracy": r("anyboxed_correct"),
         "tag_rates": {k: r(k) for k in ("has_exploration_open_at_start", "has_exploration_open", "has_exploration_close",
                                         "has_execution_open", "has_execution_close")},
         "categories": pd.Series([x["category"] for x in rows]).value_counts().to_dict()}
    json.dump(m, open(os.path.join(a.out, f"metrics_{a.name}.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(a.out, f"per_example_{a.name}.jsonl"), "w") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("T1", a.name, json.dumps({k: v for k, v in m.items() if k not in ("model",)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
