"""Gate A for the dual-init bridge pilot: verify what each run actually loads (no training).

1. Stage-1 provenance: path, run manifest, save format, shard sha256 vs the step-800 FSDP export, tensor-level diff vs the
   original Qwen/Qwen3-4B snapshot (proves the base is the SFT weights, not the raw base model).
2. Loaded model == Stage-1 safetensors (exact tensor equality on sampled base weights, through the LoRA wrapper).
3. LoRA / adapter: target modules, rank, alpha, numel, requires_grad list; A = fresh (lora_B = 0), B == joint_step50 LoRA;
   what joint_step50.pt contains.  Projector forward hook count during scoring / generation must be 0.
4. 0-step behaviour on a fixed question batch: normal explicit mode (Q -> plan -> execution); latent-template equivalence
   with the native Stage-1 string; text plan placed in the latent template; first-token distribution and greedy head under
   Z_T, zero slots and 8 real plan-token embeddings; generation KV vs full-forward logits consistency.
"""
import argparse
import glob
import hashlib
import json
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

from later.scripts.bridge.mini_bridge import correct, last_boxed
from later.scripts.stage2t.bridge_dual import normal_mode, sha16
from later.scripts.stage2t.contrastive_2b import exec_forward, generate_batch
from later.scripts.stage2t.three_step import TStudent, load_jsonl
from later.src.train.stage2_plan_dataset import PLAN_OPEN, SEPARATOR, render_prompt

BASE_SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c")
PROBE = ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.17.mlp.down_proj.weight",
         "model.layers.35.self_attn.o_proj.weight", "lm_head.weight"]


def sha256_full(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def st_tensor(d, name):
    idx = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
    if name not in idx:
        return None
    with safe_open(f"{d}/{idx[name]}", "pt") as f:
        return f.get_tensor(name)


def first_tok(S, q, Z, k=5):
    with torch.no_grad():
        _, _, lg, _ = exec_forward(S, q, Z.unsqueeze(0), [[S.eos[0]]], n_logits=1)
    p = torch.softmax(lg[0][0].float(), -1)
    top = torch.topk(p, k)
    close = S.tok.encode("</EXECUTION>", add_special_tokens=False)[0]
    return {"top": [(S.tok.decode([int(i)]), round(float(v), 4)) for v, i in zip(top.values, top.indices)], "p_close_tag_first_token": round(float(p[close]), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["A", "B"], required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--s2b", required=True); ap.add_argument("--init", default=None)
    ap.add_argument("--pairs", required=True, help="jsonl with question/answer/plan_A_norm/plan_B_norm (fixed probe batch)")
    ap.add_argument("--latents", required=True); ap.add_argument("--out", required=True); ap.add_argument("--hash_shards", type=int, default=1)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(0)
    g = {"run": a.run}
    # ---- 1. provenance
    root = os.path.dirname(a.model.rstrip("/"))
    man = json.load(open(f"{root}/run_manifest.json"))
    g["stage1"] = {"path": a.model, "model_name": man.get("model_name"), "base_model": man.get("base_model"),
                   "training_strategy": man.get("training", {}).get("strategy"), "lora_rank_in_manifest": man.get("training", {}).get("lora_rank", "see strategy"),
                   "config_architectures": json.load(open(f"{a.model}/config.json"))["architectures"],
                   "files": sorted(os.path.basename(x) for x in glob.glob(f"{a.model}/*")),
                   "adapter_files_present": bool(glob.glob(f"{a.model}/adapter_*")),
                   "global_step": "800 (best_hf = export of checkpoints/global_step_800/huggingface; best by full-val loss, report.md)"}
    if a.hash_shards:
        step800 = f"{root}/checkpoints/global_step_800/huggingface"
        g["stage1"]["shard_sha256"] = {os.path.basename(p): sha256_full(p) for p in sorted(glob.glob(f"{a.model}/*.safetensors"))}
        g["stage1"]["step800_export_shard_sha256"] = {os.path.basename(p): sha256_full(p) for p in sorted(glob.glob(f"{step800}/*.safetensors"))}
        g["stage1"]["identical_to_global_step_800_export"] = g["stage1"]["shard_sha256"] == g["stage1"]["step800_export_shard_sha256"]
    diffs = {}
    for n in PROBE:
        s1, b0 = st_tensor(a.model, n), st_tensor(BASE_SNAP, n)
        if s1 is None or b0 is None:
            diffs[n] = "missing (tied?)"; continue
        s1, b0 = s1.float(), b0.float()
        diffs[n] = {"rel_l2_diff_vs_Qwen3-4B": float((s1 - b0).norm() / b0.norm()), "frac_elements_changed": float((s1 != b0).float().mean())}
    g["stage1"]["diff_vs_original_Qwen3_4B"] = {"base_snapshot": BASE_SNAP, "tensors": diffs}
    # ---- 2. loaded model
    S = TStudent(a.model, a.s2b, seed=0)
    ck = None
    if a.init:
        ck = torch.load(a.init, weights_only=False); S.load(ck["state"])
    for _, p in S.proj:
        p.requires_grad_(False)
    eq = {}
    sd = dict(S.peft.named_parameters())
    for n in PROBE[1:4]:
        mod = n.replace("model.", "base_model.model.model.", 1).replace(".weight", ".base_layer.weight")
        if mod not in sd:
            mod = n.replace("model.", "base_model.model.model.", 1)
        eq[n] = bool(torch.equal(sd[mod].detach().cpu(), st_tensor(a.model, n)))
    eq["model.embed_tokens.weight"] = bool(torch.equal(S.base.get_input_embeddings().weight.detach().cpu(), st_tensor(a.model, "model.embed_tokens.weight")))
    g["loaded_base_equals_stage1_safetensors"] = eq
    # ---- 3. adapters
    cfg = S.peft.peft_config["default"]
    g["lora"] = {"target_modules": sorted(cfg.target_modules), "r": cfg.r, "alpha": cfg.lora_alpha, "dropout": cfg.lora_dropout,
                 "n_tensors": len(S.lora), "numel": sum(p.numel() for _, p in S.lora),
                 "lora_B_abs_sum": float(sum(p.abs().sum() for n, p in S.lora if "lora_B" in n)),
                 "lora_A_abs_mean": float(torch.cat([p.detach().abs().flatten() for n, p in S.lora if "lora_A" in n]).mean())}
    g["requires_grad"] = {"lora": sum(p.numel() for _, p in S.lora if p.requires_grad), "projector": sum(p.numel() for _, p in S.proj if p.requires_grad),
                          "other": sorted({n.split(".")[-2] for n, p in S.peft.named_parameters() if p.requires_grad and "lora_" not in n})}
    if ck:
        st = ck["state"]
        g["init_ckpt"] = {"path": a.init, "sha16": sha16(a.init), "step": ck.get("step"), "phase": ck.get("phase"), "top_keys": sorted(ck.keys()),
                          "state_groups": {k: len(v) for k, v in st.items()}, "contains_base_weights": any(k not in ("proj", "lora") for k in st),
                          "lora_equal_after_load": all(torch.equal(p.detach(), st["lora"][n].to(p.device)) for n, p in S.lora),
                          "lora_B_abs_sum_in_ckpt": float(sum(v.abs().sum() for n, v in st["lora"].items() if "lora_B" in n))}
    calls = {"n": 0}
    hk = S.base.latent_projector.register_forward_hook(lambda *_: calls.__setitem__("n", calls["n"] + 1))
    # ---- 4. behaviour
    LZ = torch.load(a.latents, weights_only=False)["Z"]; S.Zraw = LZ
    P = load_jsonl(a.pairs)
    S.peft.eval(); beh = {}
    NM = normal_mode(S, [{"record_id": p["id"], "question": p["question"], "answer": str(p["answer"])} for p in P], 3072, 16)
    beh["normal_mode"] = {"n": len(NM), "correct": sum(r["correct"] for r in NM), "format_ok": sum(r["format_ok"] for r in NM),
                          "rows": [{k: r[k] for k in ("id", "pred", "correct", "format_ok", "n_tokens")} | {"head": r["text"][:240]} for r in NM]}
    tmpl = []
    for p in P[:4]:
        native = S.tok.encode(render_prompt(S.tok, p["question"]) + PLAN_OPEN + p["plan_A_norm"] + SEPARATOR, add_special_tokens=False)
        latent_style = S.prompt_ids(p["question"]) + S.tok.encode(p["plan_A_norm"] + "\n", add_special_tokens=False) + S.T.suffix_ids
        tmpl.append({"id": p["id"][:11], "native_equals_prompt+plan+suffix": native == latent_style,
                     "prefix_decoded_tail": S.tok.decode(S.prompt_ids(p["question"])[-8:]), "suffix_tokens": S.tok.convert_ids_to_tokens(S.T.suffix_ids)})
    beh["template_equivalence"] = tmpl
    rows = []
    for p in P[:4]:
        q = p["question"]; ZA = S.teacher(p["plan_A_norm"])
        ptoks = S.tok.encode(p["plan_A_norm"], add_special_tokens=False)[:8]
        Zemb = S.base.get_input_embeddings()(torch.tensor(ptoks, device=S.dev)).float()
        r = {"id": p["id"][:11], "first_token": {"Z_T(P_A)": first_tok(S, q, ZA), "zero_slots": first_tok(S, q, torch.zeros_like(ZA)), "plan_token_embeds_8": first_tok(S, q, Zemb)}}
        # text plan inside the latent template (no slots): exact Stage-1 string
        ids = S.prompt_ids(q) + S.tok.encode(p["plan_A_norm"] + "\n", add_special_tokens=False) + S.T.suffix_ids
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = S.peft.generate(input_ids=torch.tensor([ids], device=S.dev), max_new_tokens=80, do_sample=False, pad_token_id=S.T.pad_id)
        r["text_plan_in_template_head"] = S.tok.decode(out[0, len(ids):])
        gz = generate_batch(S, [q] * 3, torch.stack([ZA, torch.zeros_like(ZA), Zemb]), 80)
        r["greedy_head"] = {"Z_T(P_A)": gz[0], "zero_slots": gz[1], "plan_token_embeds_8": gz[2]}
        # KV-cached generation first-step logits == full-forward logits at the same position
        with torch.no_grad():
            _, _, lg, _ = exec_forward(S, q, ZA.unsqueeze(0), [[S.eos[0]]], n_logits=1)
            pres = S.prompt_ids(q); full = pres + [S.T.pad_id] * 8 + S.T.suffix_ids
            emb = S.base.get_input_embeddings()(torch.tensor([full], device=S.dev)).clone(); emb[0, len(pres):len(pres) + 8] = ZA.to(emb.dtype)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = S.base.model(inputs_embeds=emb, position_ids=torch.arange(len(full), device=S.dev)[None], use_cache=True)
            lg2 = S.base.lm_head(o.last_hidden_state[:, -1]).float()
        r["kv_vs_full_first_logits_maxabs"] = float((lg[0][0] - lg2[0]).abs().max())
        r["slot_positions"] = list(range(len(pres), len(pres) + 8)); r["n_prefix_total"] = len(full)
        rows.append(r)
    beh["latent_mode"] = rows
    g["behaviour"] = beh
    hk.remove(); g["projector_forward_calls_during_all_scoring_and_generation"] = calls["n"]
    g["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    json.dump(g, open(f"{a.out}/init_audit_{a.run}.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in g.items() if k != "behaviour"}, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
