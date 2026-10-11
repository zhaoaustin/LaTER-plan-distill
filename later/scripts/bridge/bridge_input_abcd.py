"""Bridge input repair A/B/C/D mini-pilot. Only the BRIDGE INPUT representation changes between groups; the bridge
architecture (mini_bridge.Bridge), frozen Stage-1 encoder/executor, slot injection, data, seed, data order, optimizer,
lr, grad clip and lambda_rank=0.1 are identical.

Encoder token states (all groups, = the original pilot): layer-27 residual stream at the plan tokens of
  prompt(Q) + '<EXPLORATION>\\n' + plan + '\\n'  (no execution / answer).
h_null(Q): layer-27 hidden at the boundary (last token) of prompt(Q) + '<EXPLORATION>\\n' + '' + SEPARATOR, i.e. empty
  plan + the same plan->execution boundary text.
  A raw     : u = h_tok
  B std     : u = clip((h_tok - mu_train) / max(sigma_train, floor), +-5), stats over TRAIN plan tokens
  C res     : u = h_tok - h_null(Q)                       (broadcast over plan tokens)
  D res_std : u = clip((h_tok - h_null - mu_res) / max(sigma_res, floor), +-5), stats over TRAIN residual tokens
Loss (all groups): L = 1/2 [CE_full(E_A|Z_A) + CE_full(E_B|Z_B)] + 0.1 [softplus(-D_A) + softplus(-D_B)],
  D_X = s_X(Z_X) - s_X(Z_other), s = mean log p over the FIRST `span` (64) execution targets (same positions/mask for
  both Z). Full-execution margins are logged too.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import later.scripts.bridge.mini_bridge as mb
from later.scripts.bridge.mini_bridge import Bridge, Pilot, correct, last_boxed, load_pairs, method_hits, ngram_overlap
from later.src.train.stage2_plan_dataset import PLAN_OPEN, SEPARATOR, render_prompt

CLIP = 5.0
FLOOR_PCT = 5.0


class PilotABCD(Pilot):
    def __init__(self, model_path, mode, span):
        super().__init__(model_path)
        self.mode, self.span = mode, span
        self.null_cache, self.stats = {}, None

    @torch.no_grad()
    def boundary_state(self, q, plan):
        text = render_prompt(self.tok, q) + PLAN_OPEN + plan + SEPARATOR
        ids = self.tok.encode(text, add_special_tokens=False)
        out = self.model.model(input_ids=torch.tensor([ids], device=self.dev), output_hidden_states=True, use_cache=False)
        return out.hidden_states[mb.ENC_LAYER][0, -1].float(), len(ids) - 1

    def h_null(self, q):
        if q not in self.null_cache:
            self.null_cache[q] = self.boundary_state(q, "")
        return self.null_cache[q][0]

    def base_input(self, q, plan):
        h = self.plan_states(q, plan).float()
        if self.mode in ("res", "res_std"):
            h = h - self.h_null(q)
        return h

    def fit_stats(self, pairs):
        X = torch.cat([self.base_input(p["question"], p[k]) for p in pairs for k in ("plan_A", "plan_B")])
        mu, sd = X.mean(0), X.std(0)
        floor = float(np.percentile(sd[sd > 0].cpu().numpy(), FLOOR_PCT))
        self.stats = {"mu": mu, "sigma": sd, "floor": floor, "n_tokens": X.size(0),
                      "frac_dims_floored": float((sd < floor).float().mean())}
        return self.stats

    def transform(self, q, plan):
        u = self.base_input(q, plan)
        if self.mode in ("std", "res_std"):
            u = (u - self.stats["mu"]) / torch.clamp(self.stats["sigma"], min=self.stats["floor"])
            u = u.clamp(-CLIP, CLIP)
        return u

    def Z(self, bridge, q, plans):
        Hs = [self.transform(q, p) for p in plans]
        T = max(h.size(0) for h in Hs)
        H = torch.zeros(len(Hs), T, Hs[0].size(1), device=self.dev, dtype=torch.float32)
        m = torch.zeros(len(Hs), T, dtype=torch.bool, device=self.dev)
        for i, h in enumerate(Hs):
            H[i, :h.size(0)] = h
            m[i, :h.size(0)] = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return bridge(H, m).float()

    def exec_scores2(self, q, Zs, execs):
        """-> s_full [K], s_span [K] (first self.span targets), aux. Same prefix / slots / positions for all K."""
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
        emb = self.model.get_input_embeddings()(ids).clone()
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
        rank_in_seq = torch.cat([torch.arange(int(c), device=self.dev) for c in cnt])
        s_full = torch.zeros(K, device=self.dev).index_add(0, seq, lp) / cnt
        in_span = (rank_in_seq < self.span).float()
        s_span = torch.zeros(K, device=self.dev).index_add(0, seq, lp * in_span) / torch.clamp(cnt, max=self.span)
        return s_full, s_span, {"emb": emb, "slots": slots, "ids": ids, "att": att, "pos": pos, "first_logits": logits[0]}

    def pair_terms(self, bridge, p):
        Z = self.Z(bridge, p["question"], [p["plan_A"], p["plan_B"]])
        ZA, ZB = Z[0], Z[1]
        sf, ss, aux = self.exec_scores2(p["question"], torch.stack([ZA, ZB, ZB, ZA]),
                                        [p["execution_A"], p["execution_B"], p["execution_A"], p["execution_B"]])
        # index: 0 s_A(Z_A), 1 s_B(Z_B), 2 s_A(Z_B), 3 s_B(Z_A)
        l_ce = -(sf[0] + sf[1]) / 2
        mA, mB = ss[0] - ss[2], ss[1] - ss[3]
        l_rank = F.softplus(-mA) + F.softplus(-mB)
        return {"loss": l_ce + mb.LAMBDA_RANK * l_rank, "l_ce": l_ce, "l_rank": l_rank, "margin_A": mA, "margin_B": mB,
                "margin_A_full": sf[0] - sf[2], "margin_B_full": sf[1] - sf[3], "s_full": sf, "s_span": ss, "Z": Z, "aux": aux}


@torch.no_grad()
def evaluate(P, bridge, pairs):
    rows = []
    for p in pairs:
        t = P.pair_terms(bridge, p)
        sf, ss, Z = t["s_full"].tolist(), t["s_span"].tolist(), t["Z"]
        rows.append({"id": p["id"],
                     "span_s_A_ZA": ss[0], "span_s_A_ZB": ss[2], "span_s_B_ZB": ss[1], "span_s_B_ZA": ss[3],
                     "full_s_A_ZA": sf[0], "full_s_A_ZB": sf[2], "full_s_B_ZB": sf[1], "full_s_B_ZA": sf[3],
                     "margin_A": ss[0] - ss[2], "margin_B": ss[1] - ss[3],
                     "margin_A_full": sf[0] - sf[2], "margin_B_full": sf[1] - sf[3],
                     "ce_span_A_ZA": -ss[0], "ce_span_A_ZB": -ss[2], "ce_span_B_ZB": -ss[1], "ce_span_B_ZA": -ss[3],
                     "z_rms_A": float(Z[0].pow(2).mean().sqrt()), "z_rms_B": float(Z[1].pow(2).mean().sqrt()),
                     "z_cos_AB": float(F.cosine_similarity(Z[0], Z[1], dim=-1).mean())})
    def mm(k1, k2):
        v = [r[k1] for r in rows] + [r[k2] for r in rows]
        return sum(v) / len(v), sum(x > 0 for x in v) / len(v), v
    ms, fs, v = mm("margin_A", "margin_B")
    mf, ff, vf = mm("margin_A_full", "margin_B_full")
    summ = {"mean_margin_span": ms, "frac_pos_span": fs, "mean_margin_full": mf, "frac_pos_full": ff,
            "pairs_both_pos_span": sum(r["margin_A"] > 0 and r["margin_B"] > 0 for r in rows),
            "ce_full_correct": -sum(r["full_s_A_ZA"] + r["full_s_B_ZB"] for r in rows) / (2 * len(rows)),
            "ce_full_swapped": -sum(r["full_s_A_ZB"] + r["full_s_B_ZA"] for r in rows) / (2 * len(rows)),
            "ce_span_correct": -sum(r["span_s_A_ZA"] + r["span_s_B_ZB"] for r in rows) / (2 * len(rows)),
            "ce_span_swapped": -sum(r["span_s_A_ZB"] + r["span_s_B_ZA"] for r in rows) / (2 * len(rows)),
            "z_cos_AB": sum(r["z_cos_AB"] for r in rows) / len(rows),
            "z_rms": sum(r["z_rms_A"] + r["z_rms_B"] for r in rows) / (2 * len(rows))}
    return summ, rows, v


def boot(v, n=1000, seed=0):
    rng = random.Random(seed)
    bs = sorted(sum(rng.choice(v) for _ in v) / len(v) for _ in range(n))
    return [bs[int(0.025 * n)], bs[int(0.975 * n) - 1]]


def audit(P, train, held):
    """Input representation audit (no training)."""
    def stats(pairs):
        rms, clipfrac, cos_pool = [], [], []
        for p in pairs:
            ua, ub = P.transform(p["question"], p["plan_A"]), P.transform(p["question"], p["plan_B"])
            for u in (ua, ub):
                rms.append(float(u.pow(2).mean().sqrt()))
                if P.mode in ("std", "res_std"):
                    clipfrac.append(float((u.abs() >= CLIP).float().mean()))
            cos_pool.append(float(F.cosine_similarity(ua.mean(0), ub.mean(0), dim=0)))
        return {"input_rms_mean": float(np.mean(rms)), "clip_frac_mean": float(np.mean(clipfrac)) if clipfrac else 0.0,
                "cos_meanpool_uA_uB_same_q": float(np.mean(cos_pool))}
    res = {"mode": P.mode, "train": stats(train), "heldout": stats(held)}
    if P.stats:
        sd = P.stats["sigma"].cpu().numpy()
        res["stats"] = {"n_train_tokens": P.stats["n_tokens"], "sigma_floor": P.stats["floor"],
                        "frac_dims_floored": P.stats["frac_dims_floored"],
                        "sigma_percentiles_1_5_50_95_99_max": [float(x) for x in np.percentile(sd, [1, 5, 50, 95, 99, 100])]}
    # spec-literal boundary vectors: h(Q,P,b), h(Q,0,b), residual; positions
    b = []
    for p in train + held:
        hA, iA = P.boundary_state(p["question"], p["plan_A"])
        hB, iB = P.boundary_state(p["question"], p["plan_B"])
        hN, iN = P.null_cache.get(p["question"]) or P.boundary_state(p["question"], "")
        b.append({"id": p["id"], "boundary_idx_A": iA, "boundary_idx_B": iB, "boundary_idx_null": iN,
                  "plan_len_tokens_A": iA - iN, "plan_len_tokens_B": iB - iN,
                  "cos_boundary_raw_A_B": float(F.cosine_similarity(hA, hB, dim=0)),
                  "cos_boundary_residual_A_B": float(F.cosine_similarity(hA - hN, hB - hN, dim=0)),
                  "residual_rms_A": float((hA - hN).pow(2).mean().sqrt()), "raw_rms_A": float(hA.pow(2).mean().sqrt())})
    res["boundary_literal"] = {"mean_cos_raw_A_B": float(np.mean([x["cos_boundary_raw_A_B"] for x in b])),
                               "mean_cos_residual_A_B": float(np.mean([x["cos_boundary_residual_A_B"] for x in b])),
                               "per_pair": b,
                               "caveat": "h(Q,P,b) and h(Q,0,b) sit at different absolute positions (boundary_idx differ by "
                                         "the plan length) -> residual mixes plan content with RoPE position / length effects"}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["test", "train", "final", "audit"], required=True)
    ap.add_argument("--input_mode", choices=["raw", "std", "res", "res_std"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="dir with train_pairs.jsonl / heldout_pairs.jsonl")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--span", type=int, default=64)
    ap.add_argument("--pairs_per_step", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--gen_tokens", type=int, default=1024)
    a = ap.parse_args()
    torch.manual_seed(0)
    random.seed(0)
    os.makedirs(a.run_dir, exist_ok=True)
    train = load_pairs(os.path.join(a.out, "train_pairs.jsonl"))
    held = load_pairs(os.path.join(a.out, "heldout_pairs.jsonl"))
    P = PilotABCD(a.model, a.input_mode, a.span)
    d = P.model.config.hidden_size
    bridge = Bridge(d, P.tok_rms).to(P.dev)            # same seed -> identical init in every group
    if a.input_mode in ("std", "res_std"):
        st = P.fit_stats(train)                         # TRAIN pairs only
        np.savez(os.path.join(a.run_dir, "train_stats.npz"), mu=st["mu"].cpu().numpy(), sigma=st["sigma"].cpu().numpy(),
                 floor=st["floor"], clip=CLIP, floor_pct=FLOOR_PCT, n_tokens=st["n_tokens"],
                 train_ids=np.array([p["id"] for p in train]))
    if a.mode == "audit":
        json.dump(audit(P, train, held), open(os.path.join(a.run_dir, "input_representation_audit.json"), "w"), indent=1)
        print("audit done")
        return
    if a.mode == "test":
        res = {"input_mode": a.input_mode, "bridge_params": sum(p.numel() for p in bridge.parameters()),
               "executor_trainable_params": sum(p.numel() for p in P.model.parameters() if p.requires_grad)}
        snap = {k: v.detach().cpu().clone() for k, v in P.model.state_dict().items()}
        p = train[0]
        ids, _ = P.encoder_ids(p["question"], p["plan_A"])
        res["encoder_contains_execution"] = "<EXECUTION>" in P.tok.decode(ids) or p["execution_A"][:200] in P.tok.decode(ids)
        nid = P.tok.encode(render_prompt(P.tok, p["question"]) + PLAN_OPEN + SEPARATOR, add_special_tokens=False)
        res["null_reference_text_tail"] = P.tok.decode(nid[-12:])
        res["null_reference_contains_execution_text"] = p["execution_A"][:100] in P.tok.decode(nid)
        pre, _ = P.prefix_ids(p["question"])
        res["executor_prefix_has_no_plan_tokens"] = P.tok.decode(pre).find(p["plan_A"][40:120]) < 0
        opt = torch.optim.AdamW(bridge.parameters(), lr=a.lr, weight_decay=0.01)
        bridge.train()
        smoke = []
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            t = P.pair_terms(bridge, p)
            t["loss"].backward()
            g = [q.grad for q in bridge.parameters()]
            fin = all(x is not None and torch.isfinite(x).all().item() for x in g)
            gn = float(torch.sqrt(sum(x.pow(2).sum() for x in g)))
            u = P.transform(p["question"], p["plan_A"])
            aux = t["aux"]
            L0 = len(pre)
            smoke.append({"step": step, "loss": float(t["loss"]), "grad_finite": fin, "grad_norm": gn,
                          "input_rms": float(u.pow(2).mean().sqrt()), "z_rms": float(t["Z"].pow(2).mean().sqrt()),
                          "z_cos_AB": float(F.cosine_similarity(t["Z"][0], t["Z"][1], dim=-1).mean()),
                          "cross_ce_full": {"A|ZA": -float(t["s_full"][0]), "B|ZB": -float(t["s_full"][1]),
                                            "A|ZB": -float(t["s_full"][2]), "B|ZA": -float(t["s_full"][3])},
                          "slots_equal_Z": float((aux["emb"][:, aux["slots"]].float() - torch.stack(
                              [t["Z"][0], t["Z"][1], t["Z"][1], t["Z"][0]]).detach().to(torch.bfloat16).float()).abs().max()) == 0.0,
                          "prefix_ids_pos_mask_equal": bool((aux["ids"][:, :L0] == aux["ids"][0, :L0]).all() and
                                                            (aux["pos"][:, :L0] == aux["pos"][0, :L0]).all() and (aux["att"][:, :L0] == 1).all()),
                          "executor_grads_none": all(q.grad is None for q in P.model.parameters())})
            opt.step()
        res["smoke"] = smoke
        sd = P.model.state_dict()
        res["executor_bitwise_unchanged"] = all(torch.equal(sd[k].cpu(), snap[k]) for k in snap)
        # train path vs generation path: first execution-token distribution from the same Z
        bridge.eval()
        with torch.no_grad():
            Z = P.Z(bridge, p["question"], [p["plan_A"]])
            _, _, aux = P.exec_scores2(p["question"], Z, [p["execution_A"]])
            pre_ids = torch.tensor([pre], device=P.dev)
            emb = P.model.get_input_embeddings()(pre_ids).clone()
            emb[0, P.prefix_ids(p["question"])[1]] = Z[0].to(emb.dtype)
            g = P.model.generate(inputs_embeds=emb, attention_mask=torch.ones_like(pre_ids), max_new_tokens=1, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True, pad_token_id=P.T.pad_id)
            res["train_vs_generate_first_token_argmax_equal"] = int(aux["first_logits"].argmax()) == int(g.scores[0][0].argmax())
            res["train_vs_generate_first_logit_max_abs"] = float((aux["first_logits"] - g.scores[0][0].float()).abs().max())
        hard = [res["executor_trainable_params"] == 0, not res["encoder_contains_execution"],
                not res["null_reference_contains_execution_text"], res["executor_prefix_has_no_plan_tokens"],
                all(s["grad_finite"] and s["grad_norm"] > 0 and s["slots_equal_Z"] and s["prefix_ids_pos_mask_equal"]
                    and s["executor_grads_none"] and np.isfinite(s["loss"]) for s in smoke),
                res["executor_bitwise_unchanged"], res["train_vs_generate_first_token_argmax_equal"]]
        res["ALL_HARD_GATES_PASS"] = all(hard)
        json.dump(res, open(os.path.join(a.run_dir, "unit_tests_smoke.json"), "w"), indent=1)
        print(json.dumps(res, indent=1))
        return
    if a.mode == "train":
        log = open(os.path.join(a.run_dir, "train_log.jsonl"), "w")
        opt = torch.optim.AdamW(bridge.parameters(), lr=a.lr, weight_decay=0.01)
        rng = random.Random(1234)
        os.makedirs(os.path.join(a.run_dir, "bridge_checkpoint"), exist_ok=True)
        save = lambda s: torch.save({"bridge": bridge.state_dict(), "step": s, "input_mode": a.input_mode, "span": a.span},
                                    os.path.join(a.run_dir, "bridge_checkpoint", f"step{s}.pt"))
        def ev(s):
            bridge.eval()
            hs, _, _ = evaluate(P, bridge, held)
            ts, _, _ = evaluate(P, bridge, train)
            bridge.train()
            log.write(json.dumps({"step": s, "eval": True, "heldout": hs, "train": ts}) + "\n"); log.flush()
            print("EVAL %d held span %.5f (%.2f) full %.5f | train span %.5f (%.2f) | zcos %.4f | ce %.4f/%.4f" % (
                s, hs["mean_margin_span"], hs["frac_pos_span"], hs["mean_margin_full"], ts["mean_margin_span"],
                ts["frac_pos_span"], hs["z_cos_AB"], hs["ce_full_correct"], hs["ce_full_swapped"]), flush=True)
        save(0); ev(0)
        bridge.train()
        order, t0, step = [], time.time(), 0
        torch.cuda.reset_peak_memory_stats()
        while step < a.steps:
            if not order:
                idx = list(range(len(train))); rng.shuffle(idx); order = idx
            batch = [order.pop() for _ in range(min(a.pairs_per_step, len(order)))]
            opt.zero_grad(set_to_none=True)
            agg = {k: 0.0 for k in ("loss", "l_ce", "l_rank", "margin_A", "margin_B", "margin_A_full", "margin_B_full")}
            zc = 0.0
            for i in batch:
                p = train[i]
                if rng.random() < 0.5:
                    p = {**p, "plan_A": p["plan_B"], "execution_A": p["execution_B"], "plan_B": p["plan_A"], "execution_B": p["execution_A"]}
                t = P.pair_terms(bridge, p)
                (t["loss"] / len(batch)).backward()
                for k in agg:
                    agg[k] += float(t[k]) / len(batch)
                zc += float(F.cosine_similarity(t["Z"][0], t["Z"][1], dim=-1).mean()) / len(batch)
            gn = float(torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0))
            opt.step()
            step += 1
            log.write(json.dumps({"step": step, **agg, "z_cos_AB_batch": zc, "grad_norm": gn,
                                  "pairs": [train[i]["id"] for i in batch], "elapsed_s": time.time() - t0,
                                  "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}) + "\n"); log.flush()
            if step % a.eval_every == 0:
                ev(step)
            if step in (50, 100):
                save(step)
        print("train time %.1fs peak mem %.1f GB" % (time.time() - t0, torch.cuda.max_memory_allocated() / 2**30))
        return
    if a.mode == "final":
        st = torch.load(a.ckpt)
        bridge.load_state_dict(st["bridge"]); bridge.eval()
        hs, rows, v = evaluate(P, bridge, held)
        ts, _, vt = evaluate(P, bridge, train)
        gens = []
        with torch.no_grad():
            for p in held:
                Z = P.Z(bridge, p["question"], [p["plan_A"], p["plan_B"]])
                for cond, z in (("Z_A", Z[0]), ("Z_B", Z[1])):
                    g = P.generate(p["question"], Z=z, max_new_tokens=a.gen_tokens)
                    pred = last_boxed(g)
                    gens.append({"id": p["id"], "condition": cond, "generation": g, "pred": pred, "correct": correct(pred, str(p["answer"])),
                                 "overlap_E_A": ngram_overlap(g, p["execution_A"]), "overlap_E_B": ngram_overlap(g, p["execution_B"]),
                                 "method_terms_A_hits": method_hits(g, p["method_terms_only_A"]),
                                 "method_terms_B_hits": method_hits(g, p["method_terms_only_B"])})
        res = {"ckpt": a.ckpt, "step": st["step"], "input_mode": a.input_mode, "heldout": hs, "heldout_margin_span_ci95": boot(v),
               "train": ts, "heldout_rows": rows}
        tag = f"step{st['step']}"
        json.dump(res, open(os.path.join(a.run_dir, f"final_{tag}.json"), "w"), indent=1)
        with open(os.path.join(a.run_dir, f"generations_{tag}.jsonl"), "w") as f:
            for r in gens:
                f.write(json.dumps(r) + "\n")
        print(json.dumps({k: v for k, v in res.items() if k != "heldout_rows"}, indent=1))


if __name__ == "__main__":
    main()
