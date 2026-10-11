"""Mini Bridge pilot, data step 1: sample Stage-1 E2C (step 800) solutions with vLLM (n samples / question) on a
fixed slice of stage2_val, parse <EXPLORATION> plan / <EXECUTION> execution, grade the boxed answer with math_verify.
Only the model's own generations are stored; nothing is trained."""
import argparse, json, re
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--val", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--start", type=int, default=256); ap.add_argument("--n_q", type=int, default=400)
ap.add_argument("--n", type=int, default=8); ap.add_argument("--temperature", type=float, default=0.9)
ap.add_argument("--max_tokens", type=int, default=3072); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
val = pd.read_parquet(a.val).iloc[a.start:a.start + a.n_q]
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False) for q in val.question]
llm = LLM(a.model, dtype="bfloat16", seed=a.seed, gpu_memory_utilization=0.85, max_model_len=4096 + 1024)
sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=0.95, max_tokens=a.max_tokens, seed=a.seed)
outs = llm.generate(prompts, sp)

def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0: return None
    j = s.find("{", i); d = 0
    for k in range(j, len(s)):
        d += s[k] == "{"; d -= s[k] == "}"
        if d == 0: return s[j + 1:k]
    return None

from math_verify import parse, verify
def ok(pred, gold):
    if pred is None: return False
    try: return bool(verify(parse("$" + gold + "$"), parse("$" + pred + "$")))
    except Exception: return pred.replace(" ", "") == gold.replace(" ", "")

pat = re.compile(r"^<EXPLORATION>\n(.*?)\n</EXPLORATION><EXECUTION>\n(.*?)</EXECUTION>\s*$", re.S)
with open(a.out, "w") as f:
    for (_, r), o in zip(val.iterrows(), outs):
        for k, c in enumerate(o.outputs):
            m = pat.match(c.text)
            plan, exe = (m.group(1), m.group(2)) if m else (None, None)
            pred = last_boxed(exe) if exe else None
            f.write(json.dumps({"record_id": r.record_id, "question": r.question, "gold_answer": r.answer, "sample": k,
                                "finish": c.finish_reason, "well_formed": bool(m), "plan": plan, "execution": exe,
                                "pred": pred, "correct": bool(m) and ok(pred, str(r.answer))}) + "\n")
print("done")
