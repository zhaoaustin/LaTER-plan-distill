"""Stochastic latent planning pilot: noise at z1 + permutation-invariant Set-Matching KD + Method Choice + Execution CE.

Student = ONE Qwen3-4B (Stage-1 step 800) with ONE shared LoRA (r16 q/v/o) + the LaTER LatentProjector (fresh init,
output rescaled to the 4B token-embedding RMS = existing `latent_output_rms="token_embedding"` option). Planner and
executor are the same weights.

Planner (question only): chat(Q) + '<EXPLORATION>\\n' -> h_seed ; z1 = proj(h_seed) + sigma * s_train * eps ;
  z_{t+1} = proj(h(z_1..z_t)) via the KV-cached recurrence (with autograd, LoRA active, noisy z1 actually fed back).
Executor: chat(Q) + '<EXPLORATION>\\n' + [z1..z8] + '</EXPLORATION><EXECUTION>\\n' + targets, ONE full forward with
  use_cache=False (no reuse of the planner KV), identical prefix / positions for every condition.
Teacher (train-only, frozen): T(P) = Adapter2B_step50(Compressor2A_step300(P)) in the same 4B input-embedding slots.
Per question: K=2 eps -> Z_1, Z_2 ; SetKD = min over the 2 assignments of (d(Z_i,T_A)+d(Z_j,T_B))/2 with
  d = mean(((Z-T)/rms_T)^2), rms_T per dim from train-single teacher targets ; the chosen assignment also decides which
  Z gets E_A / D_A in Choice and Exec.
Choice (choice-valid pairs only): 4-grid on the method-decision span D after the shared neutral prefix C (token LCP of
  E_A / E_B): s_XY = mean log p(D_Y | Q, Zhat_X, C) ; L = (softplus(s_AB - s_AA) + softplus(s_BA - s_BB)) / 2.
Exec: (CE(E_A|Zhat_A) + CE(E_B|Zhat_B)) / 2 over execution + '</EXECUTION><|im_end|>' tokens.
Modes: gates (1 GPU), train (torchrun 2 GPUs, manual grad all-reduce), eval (1 GPU, --shard i/n).
"""
import argparse
import hashlib
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from later.scripts.bridge.mini_bridge import correct, last_boxed, method_hits, ngram_overlap
from later.scripts.bridge.bridge_input_abcd import boot
from later.scripts.stage2b.executor_2b import Adapter4B
from later.src.train.stage2_common import load_stage2_model
from later.src.train.stage2_plan_dataset import EOS, EXEC_CLOSE, render_prompt, stage2_tokens

L = 8
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", target_modules=["q_proj", "v_proj", "o_proj"])
SPAN = 24
key = lambda s: hashlib.sha1(s.encode()).hexdigest()

# manual audit (METHOD_SPAN_AUDIT.md): D_A / D_B (first 24 tokens after the token LCP) reveal the distinct method
CHOICE_VALID = {"s1val-00645", "s1val-00689", "s1val-00472", "s1val-00626", "s1val-00496", "s1val-00344", "s1val-00535", "s1val-00491",
                "s1val-00336", "s1val-00574", "s1val-00569", "s1val-00305", "s1val-00486"}
EXCLUDED = {"s1val-00471": "plans A/B are the same method (center (r,2) + point (8,0) on the circle)"}
TINY = ["s1val-00645", "s1val-00689", "s1val-00472", "s1val-00626", "s1val-00496", "s1val-00344", "s1val-00535", "s1val-00491",
        "s1val-00359", "s1val-00660", "s1val-00313", "s1val-00653"]


def short(i):
    return i[:11]


class Student:
    def __init__(self, model_path, s2b_dir, seed=0):
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.T = stage2_tokens(self.tok)
        self.dev = "cuda"
        torch.manual_seed(seed)                                   # projector init (missing keys in the Stage-1 ckpt)
        m = load_stage2_model(model_path, latent_embed_builder_no_grad=False, latent_projector_dropout=0.0,
                              dtype=torch.bfloat16, latent_output_rms="token_embedding").to(self.dev)
        m.latent_projector.float()
        self.tok_rms = float(m.latent_projector.output_rms)
        self.peft = get_peft_model(m, LoraConfig(**LORA))
        self.base = self.peft.base_model.model
        for p in self.base.latent_projector.parameters():
            p.requires_grad_(True)
        self.lora = [(n, p) for n, p in self.peft.named_parameters() if p.requires_grad and "lora_" in n]
        self.proj = list(self.base.latent_projector.named_parameters())
        assert {id(p) for _, p in self.lora} | {id(p) for _, p in self.proj} == {id(p) for p in self.peft.parameters() if p.requires_grad}
        self.peft.eval()                                          # no dropout anywhere; randomness only from eps
        self.eos = [self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.convert_tokens_to_ids("<|endoftext|>")]
        # teacher: frozen 2A latents (cache) -> frozen 2B step-50 adapter
        zc = torch.load(f"{s2b_dir}/data/z_cache.pt", weights_only=False)
        self.Zraw = zc["Z"]
        ad = Adapter4B(2560)
        ad.load_state_dict(torch.load(f"{s2b_dir}/checkpoints/step50.pt", weights_only=False, map_location="cpu")["state"]["adapter"])
        self.adapter = ad.to(self.dev).eval()
        for p in self.adapter.parameters():
            p.requires_grad_(False)

    def params(self):
        return [p for _, p in self.proj] + [p for _, p in self.lora]

    @torch.no_grad()
    def teacher(self, plan):
        return self.adapter(self.Zraw[key(plan)].to(self.dev).unsqueeze(0))[0].float()

    def prompt_ids(self, q):
        return self.tok.encode(render_prompt(self.tok, q), add_special_tokens=False) + self.T.plan_open_ids

    def target_ids(self, e):
        return self.tok.encode(e + EXEC_CLOSE + EOS, add_special_tokens=False)

    def latents(self, q, eps, sigma, s_scale):
        """eps [K, d] (rows may be zero) -> Z [K, 8, d] fp32, autograd through projector + LoRA + KV recurrence."""
        K = eps.size(0)
        pre = self.prompt_ids(q)
        ids = torch.tensor([pre] * K, device=self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.base.model(input_ids=ids, use_cache=True)
            cache, h = out.past_key_values, out.last_hidden_state[:, -1]
            zs = []
            for t in range(L):
                z = self.base.latent_projector(h).float()
                if t == 0:
                    z = z + sigma * s_scale * eps
                zs.append(z)
                if t < L - 1:
                    pos = torch.full((K, 1), len(pre) + t, device=self.dev, dtype=torch.long)
                    out = self.base.model(inputs_embeds=z.unsqueeze(1).to(torch.bfloat16), position_ids=pos, past_key_values=cache, use_cache=True)
                    cache, h = out.past_key_values, out.last_hidden_state[:, -1]
        return torch.stack(zs, 1)

    def score(self, q, Zs, targets, spans=None):
        """Zs [K,8,d]; targets list of K token lists; spans list of (a,b) target-index ranges or None.
        -> s_full [K] mean log p over all targets, s_span [K] mean over targets[a:b], aux. One forward, use_cache=False."""
        pre = self.prompt_ids(q)
        slots = list(range(len(pre), len(pre) + L))
        pre = pre + [self.T.pad_id] * L + self.T.suffix_ids
        K = len(targets)
        n = len(pre) + max(len(t) for t in targets)
        ids = torch.full((K, n), self.T.pad_id, dtype=torch.long)
        lab = torch.full((K, n), -100, dtype=torch.long)
        att = torch.zeros(K, n, dtype=torch.long)
        for i, t in enumerate(targets):
            ids[i, :len(pre) + len(t)] = torch.tensor(pre + t)
            lab[i, len(pre):len(pre) + len(t)] = torch.tensor(t)
            att[i, :len(pre) + len(t)] = 1
        ids, lab, att = ids.to(self.dev), lab.to(self.dev), att.to(self.dev)
        emb = self.base.get_input_embeddings()(ids)
        emb = torch.cat([emb[:, :slots[0]], Zs.to(emb.dtype), emb[:, slots[-1] + 1:]], 1)
        pos = torch.arange(n, device=self.dev).unsqueeze(0).expand(K, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = self.base.model(inputs_embeds=emb, attention_mask=att, position_ids=pos, use_cache=False).last_hidden_state
        tgt = lab[:, 1:]
        sel = tgt != -100
        logits = self.base.lm_head(hs[:, :-1][sel]).float()
        lp = -F.cross_entropy(logits, tgt[sel], reduction="none")
        cnt = sel.sum(1)
        seq = torch.repeat_interleave(torch.arange(K, device=self.dev), cnt)
        r = torch.cat([torch.arange(int(c), device=self.dev) for c in cnt])
        s_full = torch.zeros(K, device=self.dev).index_add(0, seq, lp) / cnt
        s_span = None
        if spans is not None:
            a = torch.tensor([s[0] for s in spans], device=self.dev)[seq]
            b = torch.tensor([s[1] for s in spans], device=self.dev)[seq]
            m = ((r >= a) & (r < b)).float()
            s_span = torch.zeros(K, device=self.dev).index_add(0, seq, lp * m) / torch.zeros(K, device=self.dev).index_add(0, seq, m)
        return s_full, s_span, {"first_logits": logits[0], "emb": emb, "slots": slots}

    @torch.no_grad()
    def generate(self, q, Zs, max_new=2048):
        """Greedy, batched over K latent conditions (same prefix length), fresh KV per call."""
        pre = self.prompt_ids(q)
        K = Zs.size(0)
        pre_full = pre + [self.T.pad_id] * L + self.T.suffix_ids
        ids = torch.tensor([pre_full] * K, device=self.dev)
        emb = self.base.get_input_embeddings()(ids).clone()
        emb[:, len(pre):len(pre) + L] = Zs.to(emb.dtype)
        n = ids.size(1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.base.model(inputs_embeds=emb, position_ids=torch.arange(n, device=self.dev).unsqueeze(0).expand(K, -1), use_cache=True)
            cache = out.past_key_values
            logits = self.base.lm_head(out.last_hidden_state[:, -1]).float()
            first = logits.clone()
            eos = torch.tensor(self.eos, device=self.dev)
            done = torch.zeros(K, dtype=torch.bool, device=self.dev)
            gen = [[] for _ in range(K)]
            for t in range(max_new):
                tk = logits.argmax(-1)
                for i in range(K):
                    if not done[i]:
                        gen[i].append(int(tk[i]))
                done = done | torch.isin(tk, eos)
                if bool(done.all()):
                    break
                out = self.base.model(input_ids=tk.unsqueeze(1), position_ids=torch.full((K, 1), n + t, device=self.dev), past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                logits = self.base.lm_head(out.last_hidden_state[:, -1]).float()
        return [self.tok.decode(g, skip_special_tokens=False) for g in gen], first

    def state(self):
        return {"proj": {n: p.detach().clone() for n, p in self.proj}, "lora": {n: p.detach().clone() for n, p in self.lora}}

    def load(self, st):
        with torch.no_grad():
            for n, p in self.proj:
                p.copy_(st["proj"][n])
            for n, p in self.lora:
                p.copy_(st["lora"][n])


def build_pairs(S, s2b_dir):
    ps = [json.loads(l) for l in open(f"{s2b_dir}/data/pairs.jsonl")]
    out = []
    for p in ps:
        a = S.tok.encode(p["execution_A"], add_special_tokens=False)
        b = S.tok.encode(p["execution_B"], add_special_tokens=False)
        k = 0
        while k < min(len(a), len(b)) and a[k] == b[k]:
            k += 1
        C, DA, DB = a[:k], a[k:k + SPAN], b[k:k + SPAN]
        tA, tB = S.target_ids(p["execution_A"]), S.target_ids(p["execution_B"])
        assert tA[:k + len(DA)] == C + DA and tB[:k + len(DB)] == C + DB, p["id"]
        q = dict(p)
        q.update({"C": C, "D_A": DA, "D_B": DB, "tgt_A": tA, "tgt_B": tB, "choice_valid": short(p["id"]) in CHOICE_VALID,
                  "excluded": EXCLUDED.get(short(p["id"])), "C_text": S.tok.decode(C), "D_A_text": S.tok.decode(DA), "D_B_text": S.tok.decode(DB),
                  "first_decision_tokens": [S.tok.decode(DA[:1]), S.tok.decode(DB[:1])]})
        out.append(q)
    return out


def teacher_stats(S, s2b_dir):
    singles = [json.loads(l) for l in open(f"{s2b_dir}/data/train_singles.jsonl")]
    with torch.no_grad():
        Zs = torch.stack([S.Zraw[s["plan_hash"]] for s in singles]).to(S.dev)
        T = torch.cat([S.adapter(Zs[i:i + 256]) for i in range(0, len(Zs), 256)]).float()
    return T.pow(2).mean((0, 1)).sqrt().clamp_min(1e-5), T.mean(0)          # [d], [8,d]


def dist_(Z, T, rms):
    return (((Z - T) / rms) ** 2).mean()


def pair_loss(S, p, eps, sigma, s_scale, rms, lam, use_choice=True):
    Z = S.latents(p["question"], eps, sigma, s_scale)
    TA, TB = S.teacher(p["plan_A_norm"]), S.teacher(p["plan_B_norm"])
    c = [[dist_(Z[i], T, rms) for T in (TA, TB)] for i in range(2)]
    cost0, cost1 = c[0][0] + c[1][1], c[0][1] + c[1][0]
    swap = bool(cost1.detach() < cost0.detach())
    ZA, ZB = (Z[1], Z[0]) if swap else (Z[0], Z[1])
    l_set = 0.5 * (cost1 if swap else cost0)
    sf, _, _ = S.score(p["question"], torch.stack([ZA, ZB]), [p["tgt_A"], p["tgt_B"]])
    l_exec = -(sf[0] + sf[1]) / 2
    C, DA, DB = p["C"], p["D_A"], p["D_B"]
    span = lambda D: (len(C), len(C) + len(D))
    _, ss, _ = S.score(p["question"], torch.stack([ZA, ZA, ZB, ZB]), [C + DA, C + DB, C + DA, C + DB], [span(DA), span(DB), span(DA), span(DB)])
    sAA, sAB, sBA, sBB = ss
    l_choice = 0.5 * (F.softplus(sAB - sAA) + F.softplus(sBA - sBB))
    w_choice = lam["choice"] if (use_choice and p["choice_valid"]) else 0.0
    loss = lam["set"] * l_set + w_choice * l_choice + lam["exec"] * l_exec
    with torch.no_grad():
        info = {"id": p["id"], "l_set": float(l_set), "l_exec": float(l_exec), "l_choice": float(l_choice), "choice_used": w_choice > 0,
                "assign": "1B2A" if swap else "1A2B", "set_gap": float((cost0 - cost1).abs() / 2), "delta_A": float(sAA - sBA), "delta_B": float(sBB - sAB),
                "d_student_12": float(dist_(Z[0], Z[1], rms)), "d_teacher_AB": float(dist_(TA, TB, rms)),
                "z1_cos_12": float(F.cosine_similarity(Z[0, 0], Z[1, 0], 0)), "z8_cos_12": float(F.cosine_similarity(Z[0, -1], Z[1, -1], 0))}
    return loss, {"set": l_set, "choice": l_choice, "exec": l_exec}, info


def noise(seed_tuple, K, d, dev):
    g = torch.Generator(device=dev)
    g.manual_seed(int(hashlib.sha1(repr(seed_tuple).encode()).hexdigest()[:12], 16))
    return torch.randn(K, d, generator=g, device=dev)


def grad_norm(ps):
    gs = [p.grad.float().norm() for p in ps if p.grad is not None]
    return float(torch.stack(gs).norm()) if gs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gates", "train", "eval"], required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--s2b", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", type=float, default=0.10); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--lam_set", type=float, default=1.0); ap.add_argument("--lam_choice", type=float, default=0.1); ap.add_argument("--lam_exec", type=float, default=1.0)
    ap.add_argument("--q_per_rank", type=int, default=2); ap.add_argument("--tiny_steps", type=int, default=25)
    ap.add_argument("--stop_at", type=int, default=25); ap.add_argument("--resume", default=None); ap.add_argument("--save_at", default="25,50,75,100,120")
    ap.add_argument("--ckpt", default=None); ap.add_argument("--shard", default="0/1"); ap.add_argument("--gen", action="store_true")
    ap.add_argument("--gen_tokens", type=int, default=2048); ap.add_argument("--tag", default="")
    ap.add_argument("--max_wall_s", type=float, default=3 * 3600)
    a = ap.parse_args()
    ddp = a.mode == "train" and "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    os.makedirs(a.out, exist_ok=True)
    random.seed(42); np.random.seed(42)
    S = Student(a.model, a.s2b, seed=0)
    pairs = build_pairs(S, a.s2b)
    rms, Tmean = teacher_stats(S, a.s2b)
    s_scale = S.tok_rms               # z1 = proj(h) is RMS-normalized to the token-embedding RMS -> train z1 RMS == this constant
    lam = {"set": a.lam_set, "choice": a.lam_choice, "exec": a.lam_exec}
    train = [p for p in pairs if p["role"] == "train_pair" and not p["excluded"]]
    held = [p for p in pairs if p["role"] == "heldout8"]
    params = S.params()
    d = 2560

    if a.mode == "gates":
        g = {}
        g["trainable"] = {"projector": sum(p.numel() for _, p in S.proj), "lora": sum(p.numel() for _, p in S.lora),
                          "lora_modules": sorted({n.split(".lora_")[0].split(".")[-1] for n, _ in S.lora})}
        g["s_train_z1_rms"] = s_scale
        p0 = next(p for p in train if short(p["id"]) == "s1val-00645")
        q = p0["question"]
        # G1 teacher targets / interface
        TA, TB = S.teacher(p0["plan_A_norm"]), S.teacher(p0["plan_B_norm"])
        tcos = []
        for p in pairs:
            A_, B_ = S.teacher(p["plan_A_norm"]), S.teacher(p["plan_B_norm"])
            tcos.append({"id": short(p["id"]), "role": p["role"], "cos_AB": float(F.cosine_similarity(A_.flatten(), B_.flatten(), 0)),
                         "d_AB": float(dist_(A_, B_, rms)), "same_key": key(p["plan_A_norm"]) == key(p["plan_B_norm"])})
        cross = [float(dist_(S.teacher(pairs[i]["plan_A_norm"]), S.teacher(pairs[(i + 1) % len(pairs)]["plan_A_norm"]), rms)) for i in range(len(pairs))]
        g["G1_teacher_pairs"] = tcos
        g["G1_teacher_d_AB_mean"] = float(np.mean([x["d_AB"] for x in tcos])); g["G1_teacher_d_crossQ_mean"] = float(np.mean(cross))
        g["G1_teacher_no_identical_AB"] = not any(x["same_key"] for x in tcos)
        g["G1_teacher_rms"] = float(TA.pow(2).mean().sqrt()); g["G1_token_rms"] = S.tok_rms
        with torch.no_grad():
            Z0 = S.latents(q, torch.zeros(2, d, device=S.dev), 0.0, s_scale)
        g["G1_student_z_rms_per_slot"] = [float(x) for x in Z0[0].pow(2).mean(-1).sqrt()]
        g["G1_student_teacher_cos_init"] = float(F.cosine_similarity(Z0[0].flatten(), TA.flatten(), 0))
        g["G1_interface"] = "both = vectors written into the same 8 input-embedding slots after chat(Q)+'<EXPLORATION>\\n', before '</EXPLORATION><EXECUTION>\\n'"
        # G2.1 determinism / noise propagation
        with torch.no_grad():
            Za = S.latents(q, torch.zeros(1, d, device=S.dev), 0.0, s_scale)
            Zb = S.latents(q, torch.zeros(1, d, device=S.dev), 0.0, s_scale)
            e = noise(("gate", 1), 2, d, S.dev)
            Zn = S.latents(q, e, a.sigma, s_scale)
        g["G2_sigma0_repeat_max_abs"] = float((Za - Zb).abs().max())
        g["G2_noise_z1_rel_diff"] = float((Zn[0, 0] - Zn[1, 0]).norm() / Zn[0, 0].norm())
        g["G2_noise_slot_rel_diff_z1_to_z8"] = [float((Zn[0, t] - Zn[1, t]).norm() / Zn[0, t].norm()) for t in range(L)]
        g["G2_noise_propagates"] = all(x > 1e-4 for x in g["G2_noise_slot_rel_diff_z1_to_z8"][1:])
        g["G2_z1_noise_rms_ratio"] = float((Zn[0, 0] - Za[0, 0]).pow(2).mean().sqrt() / Za[0, 0].pow(2).mean().sqrt())
        # recurrence gradient: d z8 / d z1
        e_leaf = e[:1].clone().requires_grad_(True)
        Zg = S.latents(q, e_leaf, a.sigma, s_scale)
        gz = torch.autograd.grad(Zg[0, -1].sum(), e_leaf, allow_unused=True)[0]   # eps enters only via z1 -> recurrence
        g["G2_dz8_dz1_norm"] = float(gz.norm()) if gz is not None else 0.0
        # G2.3 permutation invariance
        Zp = Zn.detach()
        def setkd(Z, A_, B_):
            c0 = dist_(Z[0], A_, rms) + dist_(Z[1], B_, rms); c1 = dist_(Z[0], B_, rms) + dist_(Z[1], A_, rms)
            return float(torch.minimum(c0, c1) / 2)
        base_v = setkd(Zp, TA, TB)
        g["G2_setkd_perm"] = {"base": base_v, "swap_students": setkd(Zp.flip(0), TA, TB), "swap_teachers": setkd(Zp, TB, TA)}
        g["G2_setkd_perm_invariant"] = abs(base_v - g["G2_setkd_perm"]["swap_students"]) < 1e-6 and abs(base_v - g["G2_setkd_perm"]["swap_teachers"]) < 1e-6
        Zfake = torch.stack([TB + 0.001 * torch.randn_like(TB), TA + 0.001 * torch.randn_like(TA)])
        _, _, inf_ = None, None, None
        c0 = dist_(Zfake[0], TA, rms) + dist_(Zfake[1], TB, rms); c1 = dist_(Zfake[0], TB, rms) + dist_(Zfake[1], TA, rms)
        g["G2_setkd_rematch_on_reversed_samples"] = bool(c1 < c0)
        # G2.4/5 per-loss gradients
        gr = {}
        for name in ("set", "choice", "exec"):
            for p_ in params:
                p_.grad = None
            loss, parts, info = pair_loss(S, p0, e, a.sigma, s_scale, rms, lam)
            parts[name].backward()
            gr[name] = {"projector": grad_norm([p for _, p in S.proj]), "lora": grad_norm([p for _, p in S.lora]),
                        "projector_first_layer": grad_norm([p for n, p in S.proj if n.startswith("net.0")]),
                        "finite": all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)}
        g["G2_grad_norms_per_loss"] = gr
        g["G2_grads_nonzero_finite"] = all(v["projector"] > 0 and v["lora"] > 0 and v["finite"] for v in gr.values())
        g["G2_first_info"] = info
        for p_ in params:
            p_.grad = None
        # choice-path gradient into the planner: d L_choice / d z1 through the recurrence (z2..z8 depend on z1)
        # G2.6/7 spans: C+D is a prefix of the target ids; s_AA from the short choice forward == span of the full exec forward
        with torch.no_grad():
            Zt = S.latents(q, e, a.sigma, s_scale)
            C, DA = p0["C"], p0["D_A"]
            _, ss_short, ax = S.score(q, Zt[:1], [C + DA], [(len(C), len(C) + len(DA))])
            _, ss_long, _ = S.score(q, Zt[:1], [p0["tgt_A"]], [(len(C), len(C) + len(DA))])
        g["G2_span_short_vs_long"] = [float(ss_short[0]), float(ss_long[0])]
        g["G2_span_consistent"] = abs(float(ss_short[0]) - float(ss_long[0])) < 0.05
        g["G2_prefix_text_tail"] = S.tok.decode(S.prompt_ids(q)[-12:]) + " [8 slots] " + S.tok.decode(S.T.suffix_ids)
        dec = S.tok.decode(S.prompt_ids(q))
        g["G2_no_plan_or_execution_in_prompt"] = all(p0[k][10:80] not in dec for k in ("plan_A_norm", "plan_B_norm", "execution_A", "execution_B"))
        g["G2_C_text_neutral"] = p0["C_text"]
        # G2.8 interventions recompute: zero latent changes first logits, finite
        with torch.no_grad():
            s1, _, a1 = S.score(q, Zt[:1], [p0["tgt_A"]])
            s2, _, a2 = S.score(q, torch.zeros_like(Zt[:1]), [p0["tgt_A"]])
        g["G2_zero_changes_logits"] = float((a1["first_logits"] - a2["first_logits"]).abs().max()) > 0
        g["G2_zero_finite"] = bool(torch.isfinite(a2["first_logits"]).all())
        # train path vs generation path, first token
        with torch.no_grad():
            gens, first = S.generate(q, Zt[:1], max_new=1)
        g["G2_train_vs_generate_first_argmax_equal"] = int(a1["first_logits"].argmax()) == int(first[0].argmax())
        # smoke: 3 optimizer steps on 2 train pairs, losses finite (counted in the 120-step budget)
        snap = {n: p.detach().clone() for n, p in S.peft.named_parameters() if not p.requires_grad}
        st0 = S.state()
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
        hist = []
        t0 = time.time()
        for st in range(3):
            opt.zero_grad(set_to_none=True); tot = 0
            for j, p in enumerate(train[:2]):
                loss, parts, info = pair_loss(S, p, noise(("smoke", st, j), 2, d, S.dev), a.sigma, s_scale, rms, lam)
                (loss / 2).backward(); tot += float(loss) / 2
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); hist.append(tot)
        g["G2_smoke_losses"] = hist; g["G2_smoke_finite"] = all(math.isfinite(x) for x in hist)
        g["G2_smoke_sec_per_step_2q"] = (time.time() - t0) / 3
        g["G2_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        g["G2_base_unchanged"] = all(torch.equal(p, snap[n]) for n, p in S.peft.named_parameters() if not p.requires_grad)
        # G2.10 save / reload
        path = f"{a.out}/tmp_gate.pt"
        torch.save({"state": S.state(), "opt": opt.state_dict(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, path)
        with torch.no_grad():
            l1, _, _ = pair_loss(S, train[0], noise(("reload",), 2, d, S.dev), a.sigma, s_scale, rms, lam)
        S.load(st0)
        S.load(torch.load(path, weights_only=False)["state"])
        with torch.no_grad():
            l2, _, _ = pair_loss(S, train[0], noise(("reload",), 2, d, S.dev), a.sigma, s_scale, rms, lam)
        g["G2_reload_loss"] = [float(l1), float(l2)]; g["G2_reload_ok"] = abs(float(l1) - float(l2)) < 1e-4
        os.remove(path)
        hard = [g["G1_teacher_no_identical_AB"], g["G2_sigma0_repeat_max_abs"] == 0.0, g["G2_noise_propagates"], g["G2_dz8_dz1_norm"] > 0,
                g["G2_setkd_perm_invariant"], g["G2_setkd_rematch_on_reversed_samples"], g["G2_grads_nonzero_finite"], g["G2_span_consistent"],
                g["G2_no_plan_or_execution_in_prompt"], g["G2_zero_changes_logits"], g["G2_zero_finite"], g["G2_train_vs_generate_first_argmax_equal"],
                g["G2_smoke_finite"], g["G2_base_unchanged"], g["G2_reload_ok"]]
        g["ALL_HARD_GATES_PASS"] = all(hard)
        g["hard_gate_vector"] = hard
        json.dump(g, open(f"{a.out}/gates.json", "w"), indent=1)
        with open(f"{a.out}/pairs_with_spans.jsonl", "w") as f:
            for p in pairs:
                f.write(json.dumps({k: v for k, v in p.items() if k not in ("tgt_A", "tgt_B")}) + "\n")
        print(json.dumps({k: v for k, v in g.items() if k != "G1_teacher_pairs"}, indent=1))
        return

    if a.mode == "train":
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
        rng = random.Random(42)
        order, step = [], 0
        if a.resume:
            ck = torch.load(a.resume, weights_only=False)
            S.load(ck["state"]); opt.load_state_dict(ck["opt"]); step = ck["step"]; rng.setstate(ck["rng_py"]); order = ck["order"]
        os.makedirs(f"{a.out}/checkpoints", exist_ok=True)
        log = open(f"{a.out}/metrics.jsonl", "a") if rank == 0 else None
        save_at = {int(x) for x in a.save_at.split(",")}
        def save(name):
            if rank == 0:
                torch.save({"state": S.state(), "opt": opt.state_dict(), "step": step, "rng_py": rng.getstate(), "order": order,
                            "sigma": a.sigma, "lam": lam, "lr": a.lr}, f"{a.out}/checkpoints/{name}.pt")
        if step == 0:
            save("step0")
        t0 = time.time(); torch.cuda.reset_peak_memory_stats()
        byid = {short(p["id"]): p for p in train}
        while step < a.stop_at:
            step += 1
            pool = [byid[i] for i in TINY] if step <= a.tiny_steps else train
            need = a.q_per_rank * world
            if step == a.tiny_steps + 1 and a.tiny_steps > 0:
                order = []
            if len(order) < need:
                idx = list(range(len(pool))); rng.shuffle(idx); order = idx + order
            sel = [order.pop() for _ in range(need)]
            mine = sel[rank::world]
            opt.zero_grad(set_to_none=True)
            infos = []
            for j, i in enumerate(mine):
                eps = noise(("train", step, short(pool[i]["id"])), 2, d, S.dev)
                loss, parts, info = pair_loss(S, pool[i], eps, a.sigma, s_scale, rms, lam)
                (loss / len(mine)).backward()
                info["noise_id"] = f"train:{step}:{short(pool[i]['id'])}"
                infos.append(info)
            local_gn = grad_norm(params)
            if ddp:
                for p in params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM); p.grad.div_(world)
                allinf = [None] * world; dist.all_gather_object(allinf, infos)
                allinf = [x for r in allinf for x in r]
                gns = [None] * world; dist.all_gather_object(gns, local_gn)
            else:
                allinf, gns = infos, [local_gn]
            gproj = grad_norm([p for _, p in S.proj]); glora = grad_norm([p for _, p in S.lora])
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            ident = None
            if ddp and step in (1, a.stop_at):
                fp = torch.tensor([float(sum(p.detach().double().sum() for p in params))], device=S.dev, dtype=torch.float64)
                fps = [torch.zeros_like(fp) for _ in range(world)]; dist.all_gather(fps, fp)
                ident = len({float(x) for x in fps}) == 1
            if rank == 0:
                mean = lambda k: float(np.mean([x[k] for x in allinf]))
                rec = {"step": step, "phase": "tiny" if step <= a.tiny_steps else "main", "l_set": mean("l_set"), "l_choice": mean("l_choice"),
                       "l_exec": mean("l_exec"), "delta_A": mean("delta_A"), "delta_B": mean("delta_B"),
                       "both_pos": sum(x["delta_A"] > 0 and x["delta_B"] > 0 for x in allinf), "n_q": len(allinf),
                       "frac_assign_1A2B": float(np.mean([x["assign"] == "1A2B" for x in allinf])), "set_gap": mean("set_gap"),
                       "d_student_12": mean("d_student_12"), "d_teacher_AB": mean("d_teacher_AB"), "z1_cos_12": mean("z1_cos_12"), "z8_cos_12": mean("z8_cos_12"),
                       "grad_norm_total": gn, "grad_norm_projector": gproj, "grad_norm_lora": glora, "grad_norm_local_per_rank": gns,
                       "params_identical_across_ranks": ident, "elapsed_s": time.time() - t0, "peak_mem_gb_rank0": torch.cuda.max_memory_allocated() / 2**30,
                       "per_q": allinf}
                log.write(json.dumps(rec) + "\n"); log.flush()
                print("step %d %s set %.4f choice %.4f exec %.4f dA %.4f dB %.4f both %d/%d dS12 %.4f dTAB %.4f z8cos %.4f gn %.3f (p %.3f l %.3f) %.1fs mem %.0fG" % (
                    step, rec["phase"], rec["l_set"], rec["l_choice"], rec["l_exec"], rec["delta_A"], rec["delta_B"], rec["both_pos"], rec["n_q"],
                    rec["d_student_12"], rec["d_teacher_AB"], rec["z8_cos_12"], gn, gproj, glora, rec["elapsed_s"], rec["peak_mem_gb_rank0"]), flush=True)
            if step == a.tiny_steps:
                save("after_tiny_overfit")
            if step in save_at:
                save(f"step{step}")
            if time.time() - t0 > a.max_wall_s:
                break
        save(f"step{step}")
        if ddp:
            dist.barrier(); dist.destroy_process_group()
        return

    if a.mode == "eval":
        ck = torch.load(a.ckpt, weights_only=False)
        S.load(ck["state"]); step = ck["step"]
        si, sn = map(int, a.shard.split("/"))
        evalset = [p for p in held + train]
        mine = evalset[si::sn]
        # train-only mean latent: sigma=0 student latents of the train questions, renormalized per slot
        with torch.no_grad():
            Zm = torch.stack([S.latents(p["question"], torch.zeros(1, d, device=S.dev), 0.0, s_scale)[0] for p in train]).mean(0)
            Zm = Zm / Zm.pow(2).mean(-1, keepdim=True).sqrt() * S.tok_rms
        K = 8
        rows = []
        for p in mine:
            q = p["question"]
            grp = held if p["role"] == "heldout8" else train
            donor = grp[(grp.index(p) + 1) % len(grp)]
            with torch.no_grad():
                eps = noise(("eval", short(p["id"])), K, d, S.dev)
                Zk = S.latents(q, eps, a.sigma, s_scale)                                   # [8,8,d]
                Z0 = S.latents(q, torch.zeros(1, d, device=S.dev), 0.0, s_scale)[0]
                Zx = S.latents(donor["question"], torch.zeros(1, d, device=S.dev), 0.0, s_scale)[0]
                TA, TB = S.teacher(p["plan_A_norm"]), S.teacher(p["plan_B_norm"])
                conds = [f"self_eps{k}" for k in range(K)] + ["self_sigma0", "zero", "mean", "crossq", "teacher_A", "teacher_B"]
                Zall = torch.cat([Zk, torch.stack([Z0, torch.zeros_like(Z0), Zm, Zx, TA, TB])])
                C, DA, DB = p["C"], p["D_A"], p["D_B"]
                n = Zall.size(0)
                _, sA, _ = S.score(q, Zall, [C + DA] * n, [(len(C), len(C) + len(DA))] * n)
                _, sB, _ = S.score(q, Zall, [C + DB] * n, [(len(C), len(C) + len(DB))] * n)
                pref = (sA - sB).tolist()
                # matched delta with eval samples 0,1 (assignment by teacher distance = diagnostic labeling only)
                c0 = dist_(Zk[0], TA, rms) + dist_(Zk[1], TB, rms); c1 = dist_(Zk[0], TB, rms) + dist_(Zk[1], TA, rms)
                ia, ib = (1, 0) if c1 < c0 else (0, 1)
                dA, dB = float(sA[ia] - sA[ib]), float(sB[ib] - sB[ia])
                # best-of-8 matching (all ordered pairs) for a K=8 view
                dk = torch.stack([torch.stack([dist_(Zk[k], T, rms) for T in (TA, TB)]) for k in range(K)])  # [8,2]
                nearest = ["A" if float(dk[k, 0]) < float(dk[k, 1]) else "B" for k in range(K)]
                # exec CE under conditions (E_A and E_B both)
                ce = {}
                for cname, za, zb in (("self_matched", Zk[ia], Zk[ib]), ("self_sigma0", Z0, Z0), ("zero", torch.zeros_like(Z0), torch.zeros_like(Z0)),
                                      ("mean", Zm, Zm), ("crossq", Zx, Zx), ("self_swapped", Zk[ib], Zk[ia]), ("teacher", TA, TB)):
                    sf, _, _ = S.score(q, torch.stack([za, zb]), [p["tgt_A"], p["tgt_B"]])
                    ce[cname] = [-float(sf[0]), -float(sf[1])]
                # latent diversity
                flat = Zk.flatten(1)
                cosm = F.cosine_similarity(flat.unsqueeze(1), flat.unsqueeze(0), -1)
                iu = torch.triu_indices(K, K, 1)
                div = {"pairwise_cos_mean": float(cosm[iu[0], iu[1]].mean()),
                       "pairwise_dist_mean": float(np.mean([float(dist_(Zk[i], Zk[j], rms)) for i, j in zip(iu[0].tolist(), iu[1].tolist())])),
                       "dist_to_sigma0_mean": float(np.mean([float(dist_(Zk[k], Z0, rms)) for k in range(K)])),
                       "teacher_d_AB": float(dist_(TA, TB, rms)), "min_d_to_TA": float(dk[:, 0].min()), "min_d_to_TB": float(dk[:, 1].min()),
                       "sigma0_d_TA_TB": [float(dist_(Z0, TA, rms)), float(dist_(Z0, TB, rms))]}
            r = {"id": p["id"], "role": p["role"], "choice_valid": p["choice_valid"], "step": step, "crossq_donor": donor["id"], "conds": conds,
                 "pref_A_minus_B": pref, "sA": sA.tolist(), "sB": sB.tolist(), "matched_delta_A": dA, "matched_delta_B": dB, "matched_idx": [ia, ib],
                 "nearest_teacher_per_eps": nearest, "exec_ce": ce, "diversity": div}
            if a.gen:
                gens, _ = S.generate(q, Zall, a.gen_tokens)
                r["gens"] = []
                for cname, g_ in zip(conds, gens):
                    pred = last_boxed(g_)
                    r["gens"].append({"cond": cname, "text": g_, "pred": pred, "correct": correct(pred, str(p["answer"])),
                                      "ended": any(e in g_ for e in ("<|im_end|>", "<|endoftext|>")),
                                      "ov_A": ngram_overlap(g_, p["execution_A"]), "ov_B": ngram_overlap(g_, p["execution_B"]),
                                      "terms_A": method_hits(g_, p["method_terms_only_A"]), "terms_B": method_hits(g_, p["method_terms_only_B"])})
            rows.append(r)
            print(short(p["id"]), p["role"], "pref", [round(x, 3) for x in pref[:K]], "s0 %.3f zero %.3f mean %.3f xq %.3f TA %.3f TB %.3f" % tuple(pref[K:]),
                  "dA %.4f dB %.4f" % (dA, dB), "cos %.4f" % div["pairwise_cos_mean"], flush=True)
        os.makedirs(f"{a.out}/evaluations", exist_ok=True)
        with open(f"{a.out}/evaluations/eval_step{step}{a.tag}_shard{si}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
