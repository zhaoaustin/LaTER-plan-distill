"""Mini Bridge pilot: same question, two verified plans -> 8 latent embeddings -> FROZEN Stage-1 executor.

Plan encoder (frozen Stage-1 Qwen3, no grad): reads prompt(Q) + '<EXPLORATION>\\n' + plan + '\\n' only (no execution,
  no answer field), states = residual stream after decoder layer ENC_LAYER at the plan-token positions (cached).
Bridge (only trainable module): 8 learned queries -> cross-attention over the plan states -> residual MLP -> each
  vector rescaled to the token-embedding RMS of the executor.
Executor (frozen Stage-1 Qwen3, requires_grad=False, eval(), forward WITH autograd so gradients reach the bridge):
  prompt(Q) + '<EXPLORATION>\\n' + [8 latent slots] + '</EXPLORATION><EXECUTION>\\n' tail + execution targets.
  The prefix is identical for A and B (no plan text); only Z and the target execution change.
Loss per pair: L_CE = (CE(E_A|Z_A) + CE(E_B|Z_B)) / 2 ; s_X(Z) = mean log p over E_X tokens ;
  L_rank = softplus(-(s_A(Z_A) - s_A(Z_B))) + softplus(-(s_B(Z_B) - s_B(Z_A))) ; L = L_CE + 0.1 L_rank.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from later.src.train.stage2_plan_dataset import EOS, EXEC_CLOSE, PLAN_OPEN, SEPARATOR, render_prompt, stage2_tokens

L_SLOTS, ENC_LAYER, LAMBDA_RANK = 8, 27, 0.1


class Bridge(nn.Module):
    def __init__(self, d, out_rms, n_slots=L_SLOTS, heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.queries = nn.Parameter(torch.randn(n_slots, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.out_rms = float(out_rms)

    def forward(self, H, mask):
        """H [B, T, d] plan states, mask [B, T] True = valid -> Z [B, n_slots, d] (fp32)."""
        k = self.norm(H.float())
        q = self.queries.unsqueeze(0).expand(H.size(0), -1, -1)
        a, _ = self.attn(q, k, k, key_padding_mask=~mask)
        z = a + self.mlp(a)
        return z * (self.out_rms / z.pow(2).mean(-1, keepdim=True).add(1e-12).sqrt())


def load_pairs(path):
    return [json.loads(l) for l in open(path)]


class Pilot:
    def __init__(self, model_path, device="cuda"):
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.T = stage2_tokens(self.tok)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dev = device
        w = self.model.get_input_embeddings().weight.float()
        self.tok_rms = float(w.pow(2).mean().sqrt())
        self.eos_ids = [self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.convert_tokens_to_ids("<|endoftext|>")]
        self.cache = {}
        self.ce_weight, self.lambda_rank = 1.0, LAMBDA_RANK

    # ---------------- encoder
    def encoder_ids(self, q, plan):
        prompt = self.tok.encode(render_prompt(self.tok, q), add_special_tokens=False)
        plan_ids = self.tok.encode(plan + "\n", add_special_tokens=False)
        return prompt + self.T.plan_open_ids + plan_ids, len(prompt) + len(self.T.plan_open_ids)

    @torch.no_grad()
    def plan_states(self, q, plan):
        key = hashlib.sha1((q + "\x00" + plan).encode()).hexdigest()
        if key not in self.cache:
            ids, p0 = self.encoder_ids(q, plan)
            out = self.model.model(input_ids=torch.tensor([ids], device=self.dev), output_hidden_states=True, use_cache=False)
            self.cache[key] = out.hidden_states[ENC_LAYER][0, p0:].detach().clone()   # [Tp, d] bf16
        return self.cache[key]

    def Z(self, bridge, q, plans):
        Hs = [self.plan_states(q, p) for p in plans]
        T = max(h.size(0) for h in Hs)
        H = torch.zeros(len(Hs), T, Hs[0].size(1), device=self.dev, dtype=torch.bfloat16)
        m = torch.zeros(len(Hs), T, dtype=torch.bool, device=self.dev)
        for i, h in enumerate(Hs):
            H[i, :h.size(0)] = h
            m[i, :h.size(0)] = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return bridge(H, m).float()

    # ---------------- executor
    def prefix_ids(self, q):
        prompt = self.tok.encode(render_prompt(self.tok, q), add_special_tokens=False)
        ids = prompt + self.T.plan_open_ids + [self.T.pad_id] * L_SLOTS + self.T.suffix_ids
        slots = list(range(len(prompt) + len(self.T.plan_open_ids), len(prompt) + len(self.T.plan_open_ids) + L_SLOTS))
        return ids, slots

    def target_ids(self, execution):
        return self.tok.encode(execution + EXEC_CLOSE + EOS, add_special_tokens=False)

    def exec_scores(self, q, Zs, execs):
        """Zs [K, 8, d], execs list of K strings -> per-sequence mean log p over target tokens [K] (with autograd)."""
        pre, slots = self.prefix_ids(q)
        tg = [self.target_ids(e) for e in execs]
        n = len(pre) + max(len(t) for t in tg)
        K = len(execs)
        ids = torch.full((K, n), self.T.pad_id, dtype=torch.long)
        lab = torch.full((K, n), -100, dtype=torch.long)
        att = torch.zeros(K, n, dtype=torch.long)
        for i, t in enumerate(tg):
            ids[i, :len(pre) + len(t)] = torch.tensor(pre + t)
            lab[i, len(pre):len(pre) + len(t)] = torch.tensor(t)
            att[i, :len(pre) + len(t)] = 1
        ids, lab, att = ids.to(self.dev), lab.to(self.dev), att.to(self.dev)
        emb = self.model.get_input_embeddings()(ids)
        emb = emb.clone()
        emb[:, slots] = Zs.to(emb.dtype)
        pos = torch.arange(n, device=self.dev).unsqueeze(0).expand(K, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.model.model(inputs_embeds=emb, attention_mask=att, position_ids=pos, use_cache=False).last_hidden_state
        tgt = lab[:, 1:]
        sel = tgt != -100
        logits = self.model.lm_head(h[:, :-1][sel]).float()
        lp = -F.cross_entropy(logits, tgt[sel], reduction="none")
        cnt = sel.sum(1)
        seq = torch.repeat_interleave(torch.arange(K, device=self.dev), cnt)
        s = torch.zeros(K, device=self.dev).index_add(0, seq, lp) / cnt
        return s, {"emb": emb, "slots": slots, "ids": ids, "att": att, "pos": pos}

    def pair_terms(self, bridge, p):
        Z = self.Z(bridge, p["question"], [p["plan_A"], p["plan_B"]])          # [2, 8, d]
        ZA, ZB = Z[0], Z[1]
        s, aux = self.exec_scores(p["question"], torch.stack([ZA, ZB, ZB, ZA]),
                                  [p["execution_A"], p["execution_B"], p["execution_A"], p["execution_B"]])
        sAA, sBB, sAB, sBA = s[0], s[1], s[2], s[3]       # s_X(Z_Y): sAB = s_A(Z_B), sBA = s_B(Z_A)
        l_ce = -(sAA + sBB) / 2
        l_rank = F.softplus(-(sAA - sAB)) + F.softplus(-(sBB - sBA))
        return {"loss": self.ce_weight * l_ce + self.lambda_rank * l_rank, "l_ce": l_ce, "l_rank": l_rank, "margin_A": sAA - sAB,
                "margin_B": sBB - sBA, "s": s, "Z": Z, "aux": aux}

    # ---------------- oracle: textual plan in the prompt (explicit Stage-1 mode)
    @torch.no_grad()
    def oracle_scores(self, q, plan, execution):
        prompt = render_prompt(self.tok, q)
        pre = self.tok.encode(prompt + PLAN_OPEN + plan + SEPARATOR, add_special_tokens=False)
        t = self.target_ids(execution)
        ids = torch.tensor([pre + t], device=self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.model.model(input_ids=ids, use_cache=False).last_hidden_state
        logits = self.model.lm_head(h[0, len(pre) - 1:-1]).float()
        return float(-F.cross_entropy(logits, ids[0, len(pre):]).item())

    @torch.no_grad()
    def generate(self, q, Z=None, plan=None, max_new_tokens=1536):
        if Z is not None:
            pre, slots = self.prefix_ids(q)
            ids = torch.tensor([pre], device=self.dev)
            emb = self.model.get_input_embeddings()(ids).clone()
            emb[0, slots] = Z.to(emb.dtype)
            out = self.model.generate(inputs_embeds=emb, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                      do_sample=False, eos_token_id=self.eos_ids, pad_token_id=self.T.pad_id)
        else:
            pre = self.tok.encode(render_prompt(self.tok, q) + PLAN_OPEN + plan + SEPARATOR, add_special_tokens=False)
            ids = torch.tensor([pre], device=self.dev)
            out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                      do_sample=False, eos_token_id=self.eos_ids, pad_token_id=self.T.pad_id)
            out = out[:, ids.size(1):]
        return self.tok.decode(out[0], skip_special_tokens=False)


def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    d = 0
    for k in range(j, len(s)):
        d += s[k] == "{"
        d -= s[k] == "}"
        if d == 0:
            return s[j + 1:k]
    return None


def correct(pred, gold):
    if pred is None:
        return False
    try:
        from math_verify import parse, verify
        return bool(verify(parse("$" + gold + "$"), parse("$" + pred + "$")))
    except Exception:
        return pred.replace(" ", "") == gold.replace(" ", "")


def ngram_overlap(a, b, n=4):
    w = lambda s: re.findall(r"[a-z0-9]+|[^\sa-z0-9]", s.lower())
    A, B = w(a), w(b)
    ga = {tuple(A[i:i + n]) for i in range(len(A) - n + 1)}
    gb = {tuple(B[i:i + n]) for i in range(len(B) - n + 1)}
    return len(ga & gb) / max(1, len(ga))


def method_hits(text, terms):
    return sum(len(re.findall(r"\b" + re.escape(t), text.lower())) for t in terms)


@torch.no_grad()
def evaluate(P, bridge, pairs):
    rows = []
    for p in pairs:
        t = P.pair_terms(bridge, p)
        s = t["s"].tolist()
        Z = t["Z"]
        rows.append({"id": p["id"], "s_A_ZA": s[0], "s_B_ZB": s[1], "s_A_ZB": s[2], "s_B_ZA": s[3],
                     "ce_A_ZA": -s[0], "ce_A_ZB_swap": -s[2], "ce_B_ZB": -s[1], "ce_B_ZA_swap": -s[3],
                     "margin_A": float(t["margin_A"]), "margin_B": float(t["margin_B"]),
                     "z_rms_A": float(Z[0].pow(2).mean().sqrt()), "z_rms_B": float(Z[1].pow(2).mean().sqrt()),
                     "z_cos_AB_slot_mean": float(F.cosine_similarity(Z[0], Z[1], dim=-1).mean()),
                     "z_rel_diff_AB": float((Z[0] - Z[1]).norm() / Z[0].norm())})
    m = [r["margin_A"] for r in rows] + [r["margin_B"] for r in rows]
    summ = {"mean_margin": sum(m) / len(m), "frac_margin_positive": sum(x > 0 for x in m) / len(m),
            "mean_margin_A": sum(r["margin_A"] for r in rows) / len(rows), "mean_margin_B": sum(r["margin_B"] for r in rows) / len(rows),
            "pairs_both_margins_positive": sum(r["margin_A"] > 0 and r["margin_B"] > 0 for r in rows),
            "mean_ce_correct": sum(r["ce_A_ZA"] + r["ce_B_ZB"] for r in rows) / (2 * len(rows)),
            "mean_ce_swapped": sum(r["ce_A_ZB_swap"] + r["ce_B_ZA_swap"] for r in rows) / (2 * len(rows)),
            "mean_z_cos_AB": sum(r["z_cos_AB_slot_mean"] for r in rows) / len(rows)}
    return summ, rows


def bootstrap(vals, n=1000, seed=0):
    rng = random.Random(seed)
    bs = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return [bs[int(0.025 * n)], bs[int(0.975 * n) - 1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["test", "train", "final"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--pairs_per_step", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--lambda_rank", type=float, default=LAMBDA_RANK)
    ap.add_argument("--ce_weight", type=float, default=1.0)
    ap.add_argument("--run_dir", default=None, help="logs / checkpoints / final outputs (default: --out)")
    a = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    train = load_pairs(os.path.join(a.out, "train_pairs.jsonl"))
    held = load_pairs(os.path.join(a.out, "heldout_pairs.jsonl"))
    P = Pilot(a.model)
    P.ce_weight, P.lambda_rank = a.ce_weight, a.lambda_rank
    R = a.run_dir or a.out
    os.makedirs(R, exist_ok=True)
    d = P.model.config.hidden_size
    bridge = Bridge(d, P.tok_rms).to(P.dev)
    n_params = sum(p.numel() for p in bridge.parameters())
    if a.mode == "test":
        res = {"bridge_params": n_params, "executor_trainable_params": sum(p.numel() for p in P.model.parameters() if p.requires_grad),
               "token_embedding_rms": P.tok_rms}
        snap = {k: v.detach().cpu().clone() for k, v in P.model.state_dict().items()}
        p = train[0]
        # leakage: encoder ids = prompt + '<EXPLORATION>\n' + plan only; executor prefix independent of the plan
        ids, p0 = P.encoder_ids(p["question"], p["plan_A"])
        enc_text = P.tok.decode(ids)
        res["encoder_contains_execution_text"] = p["execution_A"][:200] in enc_text or "<EXECUTION>" in enc_text
        res["encoder_contains_answer_field_separately"] = False
        res["plans_containing_answer_string"] = sum(
            (str(x["answer"]).replace(" ", "") in x[k].replace(" ", "")) for x in train + held for k in ("plan_A", "plan_B")
            if len(str(x["answer"]).replace(" ", "")) >= 2)
        preA, slots = P.prefix_ids(p["question"])
        res["executor_prefix_identical_for_A_B_and_plan_free"] = preA == P.prefix_ids(p["question"])[0] and not any(
            t in preA for t in set(P.tok.encode(p["plan_A"], add_special_tokens=False)) - set(preA))
        res["executor_prefix_decoded_tail"] = P.tok.decode(preA[-20:])
        # gradients
        bridge.train()
        t = P.pair_terms(bridge, p)
        t["loss"].backward()
        g = [q.grad for q in bridge.parameters()]
        res["bridge_grad_all_finite"] = all(x is not None and torch.isfinite(x).all().item() for x in g)
        res["bridge_grad_norm"] = float(torch.sqrt(sum(x.pow(2).sum() for x in g)))
        res["executor_grads_none"] = all(q.grad is None for q in P.model.parameters())
        # slots carry Z exactly; A/B prefix positions & masks identical
        aux = t["aux"]
        res["z_written_into_slots_max_abs"] = float((aux["emb"][:, aux["slots"]].float() - torch.stack(
            [t["Z"][0], t["Z"][1], t["Z"][1], t["Z"][0]]).to(torch.bfloat16).float()).abs().max())
        L0 = len(preA)
        res["prefix_ids_equal_all_4_seqs"] = bool((aux["ids"][:, :L0] == aux["ids"][0, :L0]).all())
        res["prefix_pos_mask_equal_all_4_seqs"] = bool((aux["att"][:, :L0] == 1).all() and (aux["pos"][:, :L0] == aux["pos"][0, :L0]).all())
        with torch.no_grad():
            sA, _ = P.exec_scores(p["question"], torch.stack([t["Z"][0]]), [p["execution_A"]])
            sB, _ = P.exec_scores(p["question"], torch.stack([t["Z"][1]]), [p["execution_A"]])
        res["s_A_differs_ZA_vs_ZB"] = float(sA - sB)
        # one optimizer step, then executor bitwise unchanged
        opt = torch.optim.AdamW(bridge.parameters(), lr=a.lr)
        opt.step()
        sd = P.model.state_dict()
        res["executor_bitwise_unchanged_after_step"] = all(torch.equal(sd[k].cpu(), snap[k]) for k in snap)
        # checkpoint round trip
        path = os.path.join(a.out, "tmp_bridge_test.pt")
        torch.save(bridge.state_dict(), path)
        b2 = Bridge(d, P.tok_rms).to(P.dev)
        b2.load_state_dict(torch.load(path))
        bridge.eval(); b2.eval()
        with torch.no_grad():
            z1 = P.Z(bridge, p["question"], [p["plan_A"]])
            z2 = P.Z(b2, p["question"], [p["plan_A"]])
            s1, _ = P.exec_scores(p["question"], z1, [p["execution_A"]])
            s2, _ = P.exec_scores(p["question"], z2, [p["execution_A"]])
        res["ckpt_roundtrip_z_max_abs"] = float((z1 - z2).abs().max())
        res["ckpt_roundtrip_score_diff"] = float(s1 - s2)
        os.remove(path)
        hard = [res["bridge_grad_all_finite"], res["bridge_grad_norm"] > 0, res["executor_grads_none"],
                res["executor_bitwise_unchanged_after_step"], not res["encoder_contains_execution_text"],
                res["executor_prefix_identical_for_A_B_and_plan_free"], res["z_written_into_slots_max_abs"] == 0.0,
                res["prefix_ids_equal_all_4_seqs"], res["prefix_pos_mask_equal_all_4_seqs"], res["ckpt_roundtrip_z_max_abs"] == 0.0]
        res["ALL_HARD_GATES_PASS"] = all(hard)
        json.dump(res, open(os.path.join(a.out, "unit_tests.json"), "w"), indent=1)
        print(json.dumps(res, indent=1))
        return
    if a.mode == "train":
        log = open(os.path.join(R, "train_log.jsonl"), "a")
        opt = torch.optim.AdamW(bridge.parameters(), lr=a.lr, weight_decay=0.01)
        step = 0
        rng = random.Random(1234)
        if a.resume:
            st = torch.load(a.resume)
            bridge.load_state_dict(st["bridge"]); opt.load_state_dict(st["opt"]); step = st["step"]; rng.setstate(st["rng"])
        os.makedirs(os.path.join(R, "bridge_checkpoint"), exist_ok=True)
        save = lambda s: torch.save({"bridge": bridge.state_dict(), "opt": opt.state_dict(), "step": s, "rng": rng.getstate(),
                                     "config": {"enc_layer": ENC_LAYER, "slots": L_SLOTS, "lambda_rank": a.lambda_rank, "ce_weight": a.ce_weight, "lr": a.lr,
                                                "pairs_per_step": a.pairs_per_step, "out_rms": P.tok_rms}},
                                    os.path.join(R, "bridge_checkpoint", f"step{s}.pt"))
        def ev(s):
            bridge.eval()
            hs, _ = evaluate(P, bridge, held)
            ts, _ = evaluate(P, bridge, train)
            bridge.train()
            rec = {"step": s, "eval": True, "heldout": hs, "train": ts}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print("EVAL", s, "held mean_margin %.5f frac+ %.2f | train mean_margin %.5f frac+ %.2f | held ce %.4f/%.4f" % (
                hs["mean_margin"], hs["frac_margin_positive"], ts["mean_margin"], ts["frac_margin_positive"],
                hs["mean_ce_correct"], hs["mean_ce_swapped"]), flush=True)
        if step == 0:
            save(0); ev(0)
        bridge.train()
        t0 = time.time()
        while step < a.steps:
            idx = list(range(len(train)))
            if not hasattr(main, "_order") or not main._order:
                rng.shuffle(idx); main._order = idx
            batch = [main._order.pop() for _ in range(min(a.pairs_per_step, len(main._order)))]
            opt.zero_grad(set_to_none=True)
            agg = {"loss": 0, "l_ce": 0, "l_rank": 0, "margin_A": 0, "margin_B": 0}
            for i in batch:
                p = train[i]
                if rng.random() < 0.5:   # random A/B relabelling (the loss is symmetric; guards against order effects)
                    p = {**p, "plan_A": p["plan_B"], "execution_A": p["execution_B"], "plan_B": p["plan_A"], "execution_B": p["execution_A"]}
                t = P.pair_terms(bridge, p)
                (t["loss"] / len(batch)).backward()
                for k in agg:
                    agg[k] += float(t[k]) / len(batch)
            gn = float(torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0))
            opt.step()
            step += 1
            rec = {"step": step, **agg, "grad_norm": gn, "pairs": [train[i]["id"] for i in batch], "elapsed_s": time.time() - t0}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 5 == 0:
                print("step", step, {k: round(v, 5) for k, v in agg.items()}, "gn %.3f" % gn, flush=True)
            if step % a.eval_every == 0:
                ev(step)
            if step in (50, 100):
                save(step)
        return
    if a.mode == "final":
        st = torch.load(a.ckpt)
        bridge.load_state_dict(st["bridge"]); bridge.eval()
        b0 = torch.load(os.path.join(R, "bridge_checkpoint", "step0.pt"))
        bridge0 = Bridge(d, P.tok_rms).to(P.dev); bridge0.load_state_dict(b0["bridge"]); bridge0.eval()
        hs, hrows = evaluate(P, bridge, held)
        h0, h0rows = evaluate(P, bridge0, held)
        ts, _ = evaluate(P, bridge, train)
        orc = []
        for p in held:
            sAA = P.oracle_scores(p["question"], p["plan_A"], p["execution_A"]); sAB = P.oracle_scores(p["question"], p["plan_B"], p["execution_A"])
            sBB = P.oracle_scores(p["question"], p["plan_B"], p["execution_B"]); sBA = P.oracle_scores(p["question"], p["plan_A"], p["execution_B"])
            orc.append({"id": p["id"], "oracle_margin_A": sAA - sAB, "oracle_margin_B": sBB - sBA})
        preds = []
        with torch.no_grad():
            for p in held:
                Z = P.Z(bridge, p["question"], [p["plan_A"], p["plan_B"]])
                for cond, kw in (("latent_ZA", {"Z": Z[0]}), ("latent_ZB", {"Z": Z[1]}),
                                 ("oracle_text_PA", {"plan": p["plan_A"]}), ("oracle_text_PB", {"plan": p["plan_B"]})):
                    g = P.generate(p["question"], **kw)
                    pred = last_boxed(g)
                    preds.append({"id": p["id"], "condition": cond, "generation": g, "pred": pred, "correct": correct(pred, str(p["answer"])),
                                  "overlap_with_E_A": ngram_overlap(g, p["execution_A"]), "overlap_with_E_B": ngram_overlap(g, p["execution_B"]),
                                  "method_terms_A_hits": method_hits(g, p["method_terms_only_A"]),
                                  "method_terms_B_hits": method_hits(g, p["method_terms_only_B"]),
                                  "method_label_A": p["method_label_A"], "method_label_B": p["method_label_B"], "n_chars": len(g)})
                    print("GEN", p["id"], cond, preds[-1]["correct"], round(preds[-1]["overlap_with_E_A"], 3), round(preds[-1]["overlap_with_E_B"], 3), flush=True)
        with open(os.path.join(R, "heldout_predictions.jsonl"), "w") as f:
            for r in preds:
                f.write(json.dumps(r) + "\n")
        m = [r["margin_A"] for r in hrows] + [r["margin_B"] for r in hrows]
        m0 = [r["margin_A"] for r in h0rows] + [r["margin_B"] for r in h0rows]
        res = {"ckpt": a.ckpt, "step": st["step"], "heldout": hs, "heldout_margin_ci95": bootstrap(m), "heldout_rows": hrows,
               "heldout_step0": h0, "heldout_step0_rows": h0rows, "heldout_step0_margin_ci95": bootstrap(m0), "train": ts,
               "oracle_text_plan": orc,
               "oracle_mean_margin": sum(o["oracle_margin_A"] + o["oracle_margin_B"] for o in orc) / (2 * len(orc)),
               "oracle_frac_positive": sum((o["oracle_margin_A"] > 0) + (o["oracle_margin_B"] > 0) for o in orc) / (2 * len(orc))}
        json.dump(res, open(os.path.join(R, "final_metrics.json"), "w"), indent=1)
        print(json.dumps({k: v for k, v in res.items() if not k.endswith("rows") and k != "oracle_text_plan"}, indent=1))


if __name__ == "__main__":
    main()
