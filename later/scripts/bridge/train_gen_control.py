"""Positive control for the generation path: greedy Z_A / Z_B generations on the 20 TRAIN questions, with the
executor LoRA step-50 adapter (where train margins reached 0.27) or without LoRA (frozen executor). Same cached Z,
same injection / generation code and auto metrics as the held-out evaluation."""
import argparse, json, os
import torch
from peft import PeftModel
from later.scripts.bridge.bridge_input_abcd import PilotABCD
from later.scripts.bridge.executor_lora_pilot import generations, gen_summary
from later.scripts.bridge.mini_bridge import load_pairs

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--run", required=True)
ap.add_argument("--lora", action="store_true"); ap.add_argument("--gen_tokens", type=int, default=1024)
a = ap.parse_args()
P = PilotABCD(a.model, "std", 64)
if a.lora:
    PeftModel.from_pretrained(P.model, os.path.join(a.run, "lora_adapter_step50")).eval()
train = load_pairs(os.path.join(a.data, "train_pairs.jsonl"))
Zc = {k: (v[0].cuda(), v[1].cuda()) for k, v in torch.load(os.path.join(a.run, "cached_Z.pt")).items()}
G = generations(P, train, Zc, a.gen_tokens)
tag = "lora_step50" if a.lora else "no_lora"
with open(os.path.join(a.run, f"train_generations_{tag}.jsonl"), "w") as f:
    for r in G:
        f.write(json.dumps(r) + "\n")
print(tag, json.dumps(gen_summary(G)))
