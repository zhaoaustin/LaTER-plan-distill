"""Stage 2A: external plan encoder (frozen Stage-1 Qwen3-4B layer-27 states, cached) -> trainable 8-slot compressor
-> Z [8, 2560] (4B interface scale) -> trainable latent_to_decoder adapter -> independent Qwen3-0.6B decoder (frozen,
+ LoRA) that sees ONLY a fixed instruction + 8 soft tokens and reconstructs the plan. Loss = plan-token CE only.

Modes: gates (engineering gates incl. tiny overfit, fresh modules), train (300 steps + evals at 0/100/200/300),
final (full ablation + paired margins + free generation + audits on a checkpoint).
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from later.scripts.data.plan_span_rules import METHOD_LEXICON, label_chars

PROMPT = "Reconstruct the high-level mathematical plan.\n"
N_SLOTS, D4B, TOK_RMS_4B = 8, 2560, 0.02195320464670658
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
key = lambda s: hashlib.sha1(s.encode()).hexdigest()


# ------------------------------------------------------------------ modules
class Compressor(nn.Module):
    """Train-only per-dim standardization (fixed buffers) -> 8 learned queries, 2 x [masked cross-attn + MLP] (pre-LN)
    -> each vector RMS-calibrated to the 4B token-embedding RMS."""

    def __init__(self, mu, sigma, floor, clip=5.0, d=D4B, n=N_SLOTS, layers=2, heads=8, mlp_mult=2):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.clamp(torch.as_tensor(sigma, dtype=torch.float32), min=float(floor)))
        self.clip = clip
        self.queries = nn.Parameter(torch.randn(n, d) * 0.02)
        self.ln_kv = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ln_q = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers)])
        self.ln_m = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(d, d * mlp_mult), nn.GELU(), nn.Linear(d * mlp_mult, d)) for _ in range(layers)])

    def forward(self, H, mask):
        x = ((H.float() - self.mu) / self.sd).clamp(-self.clip, self.clip)
        q = self.queries.unsqueeze(0).expand(H.size(0), -1, -1)
        for i in range(len(self.attn)):
            kv = self.ln_kv[i](x)
            a, _ = self.attn[i](self.ln_q[i](q), kv, kv, key_padding_mask=~mask)
            q = q + a
            q = q + self.mlp[i](self.ln_m[i](q))
        return q * (TOK_RMS_4B / q.pow(2).mean(-1, keepdim=True).add(1e-12).sqrt())


class Adapter(nn.Module):
    """latent_to_decoder: z / tok_rms_4B -> Linear(2560, d_dec) -> per-vector RMS = decoder embedding RMS."""

    def __init__(self, d_dec, dec_rms):
        super().__init__()
        self.lin = nn.Linear(D4B, d_dec)
        self.dec_rms = float(dec_rms)

    def forward(self, z):
        e = self.lin(z / TOK_RMS_4B)
        return e * (self.dec_rms / e.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt())


# ------------------------------------------------------------------ system
class System:
    def __init__(self, dec_path, data_dir, seed=42, device="cuda"):
        torch.manual_seed(seed)
        self.dev = device
        self.tok = AutoTokenizer.from_pretrained(dec_path)
        self.dec = AutoModelForCausalLM.from_pretrained(dec_path, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        for p in self.dec.parameters():
            p.requires_grad_(False)
        emb = self.dec.get_input_embeddings().weight
        self.dec_rms = float(emb.float().pow(2).mean().sqrt())
        st = np.load(f"{data_dir}/train_stats_L27.npz")
        self.comp = Compressor(st["mu"], st["sigma"], float(st["floor"])).to(device)
        self.adapter = Adapter(self.dec.config.hidden_size, self.dec_rms).to(device)
        self.peft = get_peft_model(self.dec, LoraConfig(**LORA))
        self.prompt_ids = self.tok.encode(PROMPT, add_special_tokens=False)
        self.eos = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        cache = torch.load(f"{data_dir}/plan_states_L27.pt")
        self.states = cache["states"]

    def trainable(self):
        return [p for p in self.comp.parameters() if p.requires_grad] + list(self.adapter.parameters()) + \
               [p for n, p in self.peft.named_parameters() if p.requires_grad and "lora_" in n]

    def Z(self, texts):
        Hs = [self.states[key(t)] for t in texts]
        T = max(h.size(0) for h in Hs)
        H = torch.zeros(len(Hs), T, D4B, dtype=torch.bfloat16, device=self.dev)
        m = torch.zeros(len(Hs), T, dtype=torch.bool, device=self.dev)
        for i, h in enumerate(Hs):
            H[i, :h.size(0)] = h.to(self.dev)
            m[i, :h.size(0)] = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.comp(H, m).float()

    def targets(self, text):
        return self.tok.encode(text, add_special_tokens=False) + [self.eos]

    def score(self, Z, texts, return_logits=False):
        """Z [B, 8, D4B]; texts: target plans. -> per-token log p list per sample (target tokens incl. EOS)."""
        B = len(texts)
        tg = [self.targets(t) for t in texts]
        P = len(self.prompt_ids)
        L0 = P + N_SLOTS
        n = L0 + max(len(t) for t in tg)
        ids = torch.full((B, n), self.pad, dtype=torch.long)
        lab = torch.full((B, n), -100, dtype=torch.long)
        att = torch.zeros(B, n, dtype=torch.long)
        for i, t in enumerate(tg):
            ids[i, :P] = torch.tensor(self.prompt_ids)
            ids[i, L0:L0 + len(t)] = torch.tensor(t)
            lab[i, L0:L0 + len(t)] = torch.tensor(t)
            att[i, :L0 + len(t)] = 1
        ids, lab, att = ids.to(self.dev), lab.to(self.dev), att.to(self.dev)
        emb = self.dec.get_input_embeddings()(ids).clone()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            soft = self.adapter(Z)
        emb[:, P:L0] = soft.to(emb.dtype)
        pos = torch.arange(n, device=self.dev).unsqueeze(0).expand(B, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.dec(inputs_embeds=emb, attention_mask=att, position_ids=pos, use_cache=False)
        logits = out.logits[:, :-1].float()
        tgt = lab[:, 1:]
        sel = tgt != -100
        lp = -F.cross_entropy(logits[sel], tgt[sel], reduction="none")
        cnt = sel.sum(1).tolist()
        res, o = [], 0
        for c in cnt:
            res.append(lp[o:o + c]); o += c
        if return_logits:
            return res, {"logits": logits, "sel": sel, "emb": emb, "ids": ids, "att": att, "soft": soft, "L0": L0}
        return res

    @torch.no_grad()
    def generate(self, z, max_new=320):
        P = len(self.prompt_ids)
        ids = torch.tensor([self.prompt_ids + [self.pad] * N_SLOTS], device=self.dev)
        emb = self.dec.get_input_embeddings()(ids).clone()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emb[0, P:] = self.adapter(z.unsqueeze(0))[0].to(emb.dtype)
            out = self.dec.generate(inputs_embeds=emb, attention_mask=torch.ones_like(ids), max_new_tokens=max_new, do_sample=False,
                                    eos_token_id=[self.eos], pad_token_id=self.pad, output_scores=True, return_dict_in_generate=True)
        text = self.tok.decode(out.sequences[0], skip_special_tokens=True)
        return text, out.scores[0][0].float()

    def state(self):
        return {"comp": self.comp.state_dict(), "adapter": self.adapter.state_dict(),
                "lora": {n: p.detach().clone() for n, p in self.peft.named_parameters() if "lora_" in n}}

    def load(self, st):
        self.comp.load_state_dict(st["comp"]); self.adapter.load_state_dict(st["adapter"])
        with torch.no_grad():
            for n, p in self.peft.named_parameters():
                if "lora_" in n:
                    p.copy_(st["lora"][n])


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def derange(n, seed):
    rng = random.Random(seed)
    while True:
        p = list(range(n)); rng.shuffle(p)
        if all(p[i] != i for i in range(n)):
            return p


def boot(v, n=1000, seed=0):
    rng = random.Random(seed)
    bs = sorted(sum(rng.choice(v) for _ in v) / len(v) for _ in range(n))
    return [bs[int(0.025 * n)], bs[int(0.975 * n) - 1]]


# ------------------------------------------------------------------ evaluation
class SpanTagger:
    """Frozen span rules from the earlier fast diagnostic: method / subgoal (specific) vs generic tokens."""

    def __init__(self, tok, rules_path):
        self.tok = tok
        self.gen = json.load(open(rules_path))["generic_ngrams"]

    def specific_mask(self, text):
        lab = label_chars(text, self.gen)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        m = []
        for a, b in enc["offset_mapping"]:
            m.append(any(lab[c] in ("method_specific", "subgoal_specific") for c in range(a, b) if c < len(lab)))
        return m + [False]       # EOS


@torch.no_grad()
def ablation(S, texts, tagger, Zmean, perm, bs=16, early=32):
    """CE per sample for correct / shuffled / mean / zero; full, early (first `early` tokens), specific tokens."""
    out = {c: {"full": [], "early": [], "spec": [], "spec_n": []} for c in ("correct", "shuffled", "mean", "zero")}
    masks = [tagger.specific_mask(t) for t in texts]
    for s in range(0, len(texts), bs):
        idx = list(range(s, min(s + bs, len(texts))))
        Zc = S.Z([texts[i] for i in idx])
        Zs = S.Z([texts[perm[i]] for i in idx])
        conds = {"correct": Zc, "shuffled": Zs, "mean": Zmean.unsqueeze(0).expand(len(idx), -1, -1), "zero": torch.zeros_like(Zc)}
        for c, Z in conds.items():
            lps = S.score(Z, [texts[i] for i in idx])
            for j, i in enumerate(idx):
                lp = lps[j].double()
                m = torch.tensor(masks[i][:len(lp)], device=lp.device)
                out[c]["full"].append(float(-lp.mean())); out[c]["early"].append(float(-lp[:early].mean()))
                out[c]["spec"].append(float(-lp[m].mean()) if m.any() else float("nan")); out[c]["spec_n"].append(int(m.sum()))
    summ = {}
    for c in out:
        summ[c] = {k: float(np.nanmean(out[c][k])) for k in ("full", "early", "spec")}
    for c in ("shuffled", "mean", "zero"):
        for k in ("full", "early", "spec"):
            d = [a - b for a, b in zip(out[c][k], out["correct"][k]) if not (math.isnan(a) or math.isnan(b))]
            summ[f"gain_{c}_{k}"] = {"mean": float(np.mean(d)), "ci95": boot(d), "frac_pos": float(np.mean([x > 0 for x in d])), "n": len(d)}
    return summ, out


@torch.no_grad()
def paired(S, pairs, tagger):
    rows = []
    for p in pairs:
        A, B = p["plan_A_norm"], p["plan_B_norm"]
        Z = S.Z([A, B])
        lps = S.score(torch.stack([Z[0], Z[1], Z[1], Z[0]]), [A, A, B, B])   # l(A|ZA), l(A|ZB), l(B|ZB), l(B|ZA)
        mA, mB = tagger.specific_mask(A), tagger.specific_mask(B)
        def spec(lp, m):
            m = torch.tensor(m[:len(lp)], device=lp.device); return float(lp[m].mean()) if m.any() else float("nan")
        lA_A, lA_B, lB_B, lB_A = [float(x.mean()) for x in lps]
        rows.append({"id": p["id"], "role": p["role"], "l_A_ZA": lA_A, "l_A_ZB": lA_B, "l_B_ZB": lB_B, "l_B_ZA": lB_A,
                     "delta_A": lA_A - lA_B, "delta_B": lB_B - lB_A,
                     "delta_A_spec": spec(lps[0], mA) - spec(lps[1], mA), "delta_B_spec": spec(lps[2], mB) - spec(lps[3], mB),
                     "z_cos_AB": float(F.cosine_similarity(Z[0], Z[1], dim=-1).mean())})
    def summ(role):
        rr = [r for r in rows if role is None or r["role"] == role]
        v = [r["delta_A"] for r in rr] + [r["delta_B"] for r in rr]
        vs = [x for r in rr for x in (r["delta_A_spec"], r["delta_B_spec"]) if not math.isnan(x)]
        return {"n_pairs": len(rr), "mean_delta": float(np.mean(v)), "ci95": boot(v), "frac_pos": float(np.mean([x > 0 for x in v])),
                "pairs_both_pos": sum(r["delta_A"] > 0 and r["delta_B"] > 0 for r in rr),
                "mean_delta_specific": float(np.mean(vs)) if vs else None, "z_cos_AB": float(np.mean([r["z_cos_AB"] for r in rr]))}
    return {"heldout8": summ("heldout8"), "extended20": summ("extended20"), "all28": summ(None)}, rows


@torch.no_grad()
def train_mean_Z(S, train_texts, n=512, bs=32):
    zs = [S.Z(train_texts[s:s + bs]) for s in range(0, min(n, len(train_texts)), bs)]
    m = torch.cat(zs).mean(0)
    return m * (TOK_RMS_4B / m.pow(2).mean(-1, keepdim=True).sqrt())


@torch.no_grad()
def latent_geometry(S, texts, bs=32):
    Z = torch.cat([S.Z(texts[s:s + bs]) for s in range(0, len(texts), bs)]).flatten(1)
    Zn = F.normalize(Z, dim=-1); off = ~torch.eye(len(texts), dtype=torch.bool, device=Z.device)
    C = Z - Z.mean(0)
    return {"cross_plan_cos_raw": float((Zn @ Zn.T)[off].mean()),
            "relative_cross_plan_std": float(C.pow(2).mean().sqrt() / Z.pow(2).mean().sqrt())}


def evaluate(S, step, data, tagger, out_dir):
    S.peft.eval(); S.comp.eval(); S.adapter.eval()
    Zmean = train_mean_Z(S, data["train_texts"])
    abl, per = ablation(S, data["val_texts"], tagger, Zmean, data["val_perm"])
    pr, rows = paired(S, data["pairs"], tagger)
    geo = latent_geometry(S, data["val_texts"][:200])
    res = {"step": step, "val_ablation": abl, "paired": pr, "paired_rows": rows, "latent_geometry_val": geo}
    os.makedirs(f"{out_dir}/evaluations", exist_ok=True)
    json.dump(res, open(f"{out_dir}/evaluations/val_ablation_step{step}.json", "w"), indent=1)
    json.dump(per, open(f"{out_dir}/evaluations/val_per_sample_step{step}.json", "w"))
    S.peft.train(); S.comp.train(); S.adapter.train()
    return res


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gates", "train", "final"], required=True)
    ap.add_argument("--dec", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--steps", type=int, default=300); ap.add_argument("--micro", type=int, default=8); ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    train = load_jsonl(f"{a.data}/train_plans.jsonl"); val = load_jsonl(f"{a.data}/val_plans.jsonl")
    pairs = load_jsonl(f"{a.data}/paired_plans.jsonl")
    data = {"train_texts": [r["plan_norm"] for r in train], "val_texts": [r["plan_norm"] for r in val], "pairs": pairs,
            "val_perm": derange(len(val), a.seed)}
    S = System(a.dec, a.data, seed=a.seed)
    tagger = SpanTagger(S.tok, a.rules)
    params = S.trainable()

    if a.mode == "gates":
        g = {}
        g["trainable_params"] = {"compressor": sum(p.numel() for p in S.comp.parameters()), "adapter": sum(p.numel() for p in S.adapter.parameters()),
                                 "decoder_lora": sum(p.numel() for n, p in S.peft.named_parameters() if p.requires_grad)}
        g["decoder_nonlora_trainable"] = sum(p.numel() for n, p in S.peft.named_parameters() if p.requires_grad and "lora_" not in n)
        g["train_val_no_question_field"] = all("question" not in r for r in train + val)
        base_snap = {n: p.detach().clone() for n, p in S.peft.named_parameters() if "lora_" not in n}
        t = data["train_texts"][:4]
        # 1 no question leakage: decoder prefix ids identical for all samples = fixed prompt only
        _, aux = S.score(S.Z(t), t, return_logits=True)
        L0 = aux["L0"]; P = len(S.prompt_ids)
        g["G1_prefix_is_fixed_prompt_for_all"] = bool((aux["ids"][:, :P] == torch.tensor(S.prompt_ids, device=S.dev)).all())
        g["G1_prompt_text"] = PROMPT
        g["G1_soft_slots_written"] = float((aux["emb"][:, P:L0].float() - aux["soft"].to(aux["emb"].dtype).float()).abs().max()) == 0.0
        # 2 no future leakage
        with torch.no_grad():
            x = data["train_texts"][0]; ids = S.targets(x); half = len(ids) // 2
            alt = S.tok.decode(ids[:half]) ; z = S.Z([x])
            lp1, a1 = S.score(z, [x], return_logits=True)
            mod_ids = ids[:half] + [(i + 17) % 1000 + 200 for i in ids[half:-1]] + [S.eos]
            # build a modified sequence with the same length directly
            n = L0 + len(ids)
            def logits_for(tids):
                idt = torch.tensor([S.prompt_ids + [S.pad] * N_SLOTS + tids], device=S.dev)
                emb = S.dec.get_input_embeddings()(idt).clone()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    emb[0, P:L0] = S.adapter(z)[0].to(emb.dtype)
                    return S.dec(inputs_embeds=emb, use_cache=False).logits[0].float()
            la, lb = logits_for(ids), logits_for(mod_ids)
            g["G2_first_half_logits_max_abs_diff"] = float((la[:L0 + half] - lb[:L0 + half]).abs().max())
            g["G2_second_half_logits_differ"] = float((la[L0 + half + 1:] - lb[L0 + half + 1:]).abs().max()) > 0
        # 3 latent-only: swapping Z changes logits; generation first-step logits == teacher-forced first position
        with torch.no_grad():
            za, zb = S.Z([data["val_texts"][0]]), S.Z([data["val_texts"][1]])
            _, xa = S.score(za, [data["val_texts"][0]], return_logits=True)
            _, xb = S.score(zb, [data["val_texts"][0]], return_logits=True)
            g["G3_swap_Z_changes_logits"] = float((xa["logits"] - xb["logits"]).abs().max()) > 0
            _, s0 = S.generate(za[0], max_new=1)
            g["G3_generate_first_step_vs_teacher_forced_max_abs"] = float((s0 - xa["logits"][0, L0 - 1]).abs().max())
            g["G3_generate_first_step_argmax_equal"] = int(s0.argmax()) == int(xa["logits"][0, L0 - 1].argmax())
        # 6 first-token CE / shift / pad / eos
        with torch.no_grad():
            x2 = data["train_texts"][:2]
            lps, ax = S.score(S.Z(x2), x2, return_logits=True)
            first = S.targets(x2[0])[0]
            ce_first_manual = float(F.cross_entropy(ax["logits"][0, L0 - 1].unsqueeze(0), torch.tensor([first], device=S.dev)))
            g["G6_first_token_ce_manual_vs_score"] = [ce_first_manual, float(-lps[0][0])]
            g["G6_first_token_shift_ok"] = abs(ce_first_manual + float(lps[0][0])) < 1e-4
            g["G6_n_targets_equals_len_plus_eos"] = [len(lps[i]) == len(S.targets(x2[i])) for i in range(2)]
            g["G6_last_target_is_eos"] = all(S.targets(x)[-1] == S.eos for x in x2)
            g["G6_pad_positions_ignored"] = int(ax["sel"][1].sum()) == len(S.targets(x2[1]))
        # 5 gradient
        S.comp.train(); S.adapter.train(); S.peft.train()
        z = S.Z(t); z.retain_grad()
        loss = torch.cat(S.score(z, t)).neg().mean()
        loss.backward()
        g["G5_grad_Z_norm"] = float(z.grad.norm())
        gq = [p.grad for p in S.comp.parameters()] + [p.grad for p in S.adapter.parameters()]
        g["G5_comp_adapter_grads_finite_nonzero"] = all(x is not None and torch.isfinite(x).all().item() for x in gq) and \
            float(torch.sqrt(sum(x.pow(2).sum() for x in gq))) > 0
        gl = [p.grad for n, p in S.peft.named_parameters() if "lora_B" in n]
        g["G5_lora_B_grads_finite_nonzero"] = all(x is not None and torch.isfinite(x).all().item() for x in gl) and \
            float(torch.sqrt(sum(x.pow(2).sum() for x in gl))) > 0
        g["G5_decoder_base_grads_none"] = all(p.grad is None for n, p in S.peft.named_parameters() if "lora_" not in n)
        for p in params:
            p.grad = None
        # 7 tiny overfit (16 train plans, 40 steps) on these fresh modules
        tiny = data["train_texts"][:16]; tperm = derange(16, a.seed)
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
        def tiny_eval():
            with torch.no_grad():
                c = [float(-x.mean()) for x in S.score(S.Z(tiny), tiny)]
                s = [float(-x.mean()) for x in S.score(S.Z([tiny[i] for i in tperm]), tiny)]
            return c, s
        c0, s0_ = tiny_eval()
        hist = []
        for st in range(40):
            opt.zero_grad(set_to_none=True)
            loss = torch.cat(S.score(S.Z(tiny), tiny)).neg().mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            hist.append(float(loss))
        c1, s1 = tiny_eval()
        gap = [b - a_ for a_, b in zip(c1, s1)]
        g["G7_tiny"] = {"loss_first_last": [hist[0], hist[-1]], "ce_correct_before_after": [float(np.mean(c0)), float(np.mean(c1))],
                        "ce_shuffled_after": float(np.mean(s1)), "gap_shuffled_minus_correct_mean": float(np.mean(gap)),
                        "gap_positive": f"{sum(x > 0 for x in gap)}/16"}
        g["G7_pass_rule"] = "loss decreases AND mean gap > 0.1 nats AND >= 14/16 positive (pre-registered)"
        g["G7_pass"] = hist[-1] < hist[0] and float(np.mean(gap)) > 0.1 and sum(x > 0 for x in gap) >= 14
        # 4 frozen decoder base + optimizer membership
        g["G4_decoder_base_bitwise_unchanged"] = all(torch.equal(p, base_snap[n]) for n, p in S.peft.named_parameters() if "lora_" not in n)
        opt_ids = {id(p) for gr in opt.param_groups for p in gr["params"]}
        g["G4_optimizer_only_comp_adapter_lora"] = opt_ids == {id(p) for p in params}
        g["G4_4B_never_loaded_in_training_process"] = True
        g["G4_4B_unchanged_during_caching"] = json.load(open(f"{a.data}/cache_manifest.json"))["model_unchanged"]
        # 8 save / reload (modules + optimizer + RNG + step) -> identical loss and identical next update
        ck = {"state": S.state(), "opt": opt.state_dict(), "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), "step": 40}
        path = f"{a.out}/tmp_gate_ckpt.pt"; torch.save(ck, path)
        with torch.no_grad():
            l_before = float(torch.cat(S.score(S.Z(tiny[:4]), tiny[:4])).neg().mean())
        opt.zero_grad(set_to_none=True); torch.cat(S.score(S.Z(tiny), tiny)).neg().mean().backward(); opt.step()
        after_A = {n: p.detach().clone() for n, p in S.peft.named_parameters() if "lora_B" in n}
        ck2 = torch.load(path); S.load(ck2["state"]); opt.load_state_dict(ck2["opt"])
        with torch.no_grad():
            l_reload = float(torch.cat(S.score(S.Z(tiny[:4]), tiny[:4])).neg().mean())
        opt.zero_grad(set_to_none=True); torch.cat(S.score(S.Z(tiny), tiny)).neg().mean().backward(); opt.step()
        after_B = {n: p.detach().clone() for n, p in S.peft.named_parameters() if "lora_B" in n}
        g["G8_reload_loss_equal"] = l_before == l_reload
        g["G8_next_update_max_abs_diff"] = float(max((after_A[n] - after_B[n]).abs().max() for n in after_A))
        os.remove(path)
        hard = [g["decoder_nonlora_trainable"] == 0, g["train_val_no_question_field"], g["G1_prefix_is_fixed_prompt_for_all"], g["G1_soft_slots_written"],
                g["G2_first_half_logits_max_abs_diff"] == 0.0, g["G3_swap_Z_changes_logits"], g["G3_generate_first_step_argmax_equal"],
                g["G6_first_token_shift_ok"], all(g["G6_n_targets_equals_len_plus_eos"]), g["G6_last_target_is_eos"], g["G6_pad_positions_ignored"],
                g["G5_grad_Z_norm"] > 0, g["G5_comp_adapter_grads_finite_nonzero"], g["G5_lora_B_grads_finite_nonzero"], g["G5_decoder_base_grads_none"],
                g["G7_pass"], g["G4_decoder_base_bitwise_unchanged"], g["G4_optimizer_only_comp_adapter_lora"], g["G4_4B_unchanged_during_caching"],
                g["G8_reload_loss_equal"], g["G8_next_update_max_abs_diff"] < 1e-6]
        g["ALL_GATES_PASS"] = all(hard)
        json.dump(g, open(f"{a.out}/sanity_gates.json", "w"), indent=1)
        print(json.dumps(g, indent=1))
        return

    if a.mode == "train":
        log = open(f"{a.out}/train_log.jsonl", "w")
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))
        rng = random.Random(a.seed)
        order = []
        os.makedirs(f"{a.out}/checkpoints", exist_ok=True)
        def save(step):
            torch.save({"state": S.state(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step, "rng_py": rng.getstate(),
                        "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(), "order": list(order)},
                       f"{a.out}/checkpoints/step{step}.pt")
        base_snap = {n: p.detach().clone() for n, p in S.peft.named_parameters() if "lora_" not in n}
        save(0)
        r0 = evaluate(S, 0, data, tagger, a.out)
        print("EVAL 0", json.dumps({k: r0["val_ablation"][k] for k in ("correct", "shuffled", "mean", "zero")}), flush=True)
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        S.comp.train(); S.adapter.train(); S.peft.train()
        for step in range(1, a.steps + 1):
            opt.zero_grad(set_to_none=True)
            tot = 0.0; ntok = 0
            for _ in range(a.accum):
                if len(order) < a.micro:
                    idx = list(range(len(train))); rng.shuffle(idx); order = idx + order
                mb = [order.pop() for _ in range(a.micro)]
                txt = [data["train_texts"][i] for i in mb]
                lps = S.score(S.Z(txt), txt)
                loss = torch.cat(lps).neg().mean()          # token-mean CE within micro-batch
                (loss / a.accum).backward()
                tot += float(loss) / a.accum
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step(); sched.step()
            rec = {"step": step, "train_ce": tot, "grad_norm": gn, "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 10 == 0:
                print("step %d ce %.4f gn %.3f %.2fs/step mem %.1fGB" % (step, tot, gn, (time.time() - t0) / step, rec["peak_mem_gb"]), flush=True)
            if step in (100, 200, 300):
                save(step)
                r = evaluate(S, step, data, tagger, a.out)
                v = r["val_ablation"]
                print("EVAL %d correct %.4f shuffled %.4f mean %.4f zero %.4f | gain_shuf %.4f %s pos %.2f | paired8 %.4f %s both %d/8 | all28 %.4f both %d/28" % (
                    step, v["correct"]["full"], v["shuffled"]["full"], v["mean"]["full"], v["zero"]["full"], v["gain_shuffled_full"]["mean"],
                    [round(x, 4) for x in v["gain_shuffled_full"]["ci95"]], v["gain_shuffled_full"]["frac_pos"], r["paired"]["heldout8"]["mean_delta"],
                    [round(x, 4) for x in r["paired"]["heldout8"]["ci95"]], r["paired"]["heldout8"]["pairs_both_pos"],
                    r["paired"]["all28"]["mean_delta"], r["paired"]["all28"]["pairs_both_pos"]), flush=True)
                S.comp.train(); S.adapter.train(); S.peft.train()
        post = {"train_time_s": time.time() - t0, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                "decoder_base_bitwise_unchanged": all(torch.equal(p, base_snap[n]) for n, p in S.peft.named_parameters() if "lora_" not in n)}
        json.dump(post, open(f"{a.out}/post_training_checks.json", "w"), indent=1)
        print("POST", json.dumps(post))
        return

    if a.mode == "final":
        ck = torch.load(a.ckpt); S.load(ck["state"]); step = ck["step"]
        S.comp.eval(); S.adapter.eval(); S.peft.eval()
        gens = {"paired": [], "val": [], "paraphrase": []}
        with torch.no_grad():
            Zmean = train_mean_Z(S, data["train_texts"])
            for p in data["pairs"]:
                Z = S.Z([p["plan_A_norm"], p["plan_B_norm"]])
                for c, z in (("Z_A", Z[0]), ("Z_B", Z[1])):
                    txt, _ = S.generate(z)
                    gens["paired"].append({"id": p["id"], "role": p["role"], "condition": c, "generation": txt})
            vt = data["val_texts"][:40]; perm = derange(40, a.seed)
            Zv = torch.cat([S.Z(vt[s:s + 20]) for s in range(0, 40, 20)])
            for i in range(40):
                for c, z in (("correct", Zv[i]), ("shuffled", Zv[perm[i]])) + ((("mean", Zmean), ("zero", torch.zeros_like(Zmean))) if i < 10 else ()):
                    txt, _ = S.generate(z)
                    gens["val"].append({"val_index": i, "condition": c, "generation": txt, "gold": vt[i]})
            for r in load_jsonl(f"{a.data}/paraphrases.jsonl"):
                p = next(x for x in data["pairs"] if x["id"] == r["id"])
                zo, zp = S.Z([p["plan_A_norm"]]), S.Z([r["paraphrase"]])
                zother = S.Z([p["plan_B_norm"]])
                lp = S.score(torch.cat([zo, zp, zother]), [p["plan_A_norm"]] * 3)
                txt, _ = S.generate(zp[0])
                gens["paraphrase"].append({"id": r["id"], "method": r["method"], "generation_from_paraphrase_Z": txt,
                                           "l_orig_given_Z_orig": float(lp[0].mean()), "l_orig_given_Z_paraphrase": float(lp[1].mean()),
                                           "l_orig_given_Z_otherplan": float(lp[2].mean()),
                                           "cos_Z_orig_Z_para": float(F.cosine_similarity(zo[0], zp[0], dim=-1).mean()),
                                           "cos_Z_orig_Z_other": float(F.cosine_similarity(zo[0], zother[0], dim=-1).mean())})
        os.makedirs(f"{a.out}/evaluations", exist_ok=True)
        json.dump(gens, open(f"{a.out}/evaluations/generations_step{step}.json", "w"), indent=1)
        print("final done", step)


if __name__ == "__main__":
    main()
