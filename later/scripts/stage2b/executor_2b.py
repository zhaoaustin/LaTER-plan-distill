"""Stage 2B: frozen Stage-2A Z (cached) -> trainable 4B input adapter -> Stage-1 Qwen3-4B executor (+ LoRA).

Executor input (= the Stage-2 student layout used by all earlier latent/bridge pilots):
  chat_template(Q) + '<EXPLORATION>\\n' + [8 slots <- adapter(Z)] + '</EXPLORATION><EXECUTION>\\n' tail + E targets.
No plan text, label, execution or answer is visible in the executor prompt.
Loss per optimizer step (each rank): 4 single (Q,P,E) samples -> execution CE; 1 verified pair -> CE(E_A|Z_A), CE(E_B|Z_B)
  + ranking softplus(-D_A)+softplus(-D_B), D on the first 64 execution targets (same implementation as the bridge / LoRA
  pilots). L = mean over the 6 CE terms + 0.1 * rank. Two ranks: gradients all-reduced (mean) manually; global batch =
  8 singles + 2 pairs per optimizer step.
Modes: gates (single GPU), train (torchrun --nproc_per_node 2), eval (single GPU, a saved checkpoint).
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from later.scripts.bridge.bridge_input_abcd import PilotABCD, boot
from later.scripts.bridge.mini_bridge import correct, last_boxed, method_hits, ngram_overlap

LORA = dict(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", target_modules=["q_proj", "v_proj", "o_proj"])
TOK_RMS_4B = 0.02195320464670658
LAMBDA_RANK = 0.1


class Adapter4B(nn.Module):
    """Norm + Linear(d->d, identity init) + learnable scale (init = 4B token-embedding RMS)."""

    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.lin = nn.Linear(d, d)
        with torch.no_grad():
            self.lin.weight.copy_(torch.eye(d)); self.lin.bias.zero_()
        self.log_scale = nn.Parameter(torch.tensor(math.log(TOK_RMS_4B)))

    def forward(self, z):
        return self.lin(self.ln(z.float())) * self.log_scale.exp()


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


class Exec2B:
    def __init__(self, model_path, data_dir):
        self.P = PilotABCD(model_path, "raw", 64)            # frozen Stage-1 4B, prefix / targets / scoring / generation
        self.dev = self.P.dev
        zc = torch.load(f"{data_dir}/z_cache.pt", weights_only=False)
        self.Z = {k: v.to(self.dev) for k, v in zc["Z"].items()}
        self.Zmean = zc["Z_mean_train"].to(self.dev)
        self.adapter = Adapter4B(self.P.model.config.hidden_size).to(self.dev)
        self.peft = get_peft_model(self.P.model, LoraConfig(**LORA))
        self.lora = [(n, p) for n, p in self.peft.named_parameters() if p.requires_grad]
        assert all("lora_" in n for n, _ in self.lora)

    def params(self):
        return list(self.adapter.parameters()) + [p for _, p in self.lora]

    def soft(self, Zs):
        return self.adapter(Zs)

    def scores(self, q, Zs, execs):
        """-> s_full, s_span, aux ; Zs [K, 8, d] raw Stage-2A latents (adapter applied here, same path for all conditions)."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emb = self.soft(Zs)
        return self.P.exec_scores2(q, emb, execs)

    def generate(self, q, z, max_new=2048):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            e = self.soft(z.unsqueeze(0))[0]
        return self.P.generate(q, Z=e, max_new_tokens=max_new)

    def state(self):
        return {"adapter": self.adapter.state_dict(), "lora": {n: p.detach().clone() for n, p in self.lora}}

    def load(self, st):
        self.adapter.load_state_dict(st["adapter"])
        with torch.no_grad():
            for n, p in self.lora:
                p.copy_(st["lora"][n])


def single_loss(E, s):
    sf, ss, aux = E.scores(s["question"], E.Z[s["plan_hash"]].unsqueeze(0), [s["execution"]])
    return -sf[0], aux


def pair_terms(E, p, swap=False):
    from later.scripts.stage2a.plan_ae import key
    ZA, ZB = E.Z[key(p["plan_A_norm"])], E.Z[key(p["plan_B_norm"])]
    eA, eB = p["execution_A"], p["execution_B"]
    if swap:
        ZA, ZB, eA, eB = ZB, ZA, eB, eA
    sf, ss, aux = E.scores(p["question"], torch.stack([ZA, ZB, ZB, ZA]), [eA, eB, eA, eB])
    return {"ce": [-sf[0], -sf[1]], "rank": F.softplus(-(ss[0] - ss[2])) + F.softplus(-(ss[1] - ss[3])),
            "margin_span": [float(ss[0] - ss[2]), float(ss[1] - ss[3])], "margin_full": [float(sf[0] - sf[2]), float(sf[1] - sf[3])], "aux": aux}


@torch.no_grad()
def eval_pairs(E, pairs, cross_perm):
    from later.scripts.stage2a.plan_ae import key
    rows = []
    for i, p in enumerate(pairs):
        ZA, ZB = E.Z[key(p["plan_A_norm"])], E.Z[key(p["plan_B_norm"])]
        d = pairs[cross_perm[i]]
        dA, dB = E.Z[key(d["plan_A_norm"])], E.Z[key(d["plan_B_norm"])]
        conds = [("A", "correct", ZA), ("A", "swapped", ZB), ("A", "crossq", dA), ("A", "mean", E.Zmean), ("A", "zero", torch.zeros_like(ZA)),
                 ("B", "correct", ZB), ("B", "swapped", ZA), ("B", "crossq", dB), ("B", "mean", E.Zmean), ("B", "zero", torch.zeros_like(ZB))]
        r = {"id": p["id"], "role": p["role"], "crossq_donor": d["id"]}
        for side in ("A", "B"):
            cs = [c for c in conds if c[0] == side]
            sf, ss, _ = E.scores(p["question"], torch.stack([c[2] for c in cs]), [p["execution_" + side]] * len(cs))
            for k, (_, c, _) in enumerate(cs):
                r[f"full_s_{side}_{c}"] = float(sf[k]); r[f"span_s_{side}_{c}"] = float(ss[k])
        for kind in ("full", "span"):
            for c in ("swapped", "crossq", "mean", "zero"):
                r[f"{kind}_delta_A_vs_{c}"] = r[f"{kind}_s_A_correct"] - r[f"{kind}_s_A_{c}"]
                r[f"{kind}_delta_B_vs_{c}"] = r[f"{kind}_s_B_correct"] - r[f"{kind}_s_B_{c}"]
        rows.append(r)
    summ = {}
    for kind in ("full", "span"):
        for c in ("swapped", "crossq", "mean", "zero"):
            v = [x for r in rows for x in (r[f"{kind}_delta_A_vs_{c}"], r[f"{kind}_delta_B_vs_{c}"])]
            # bootstrap over QUESTIONS: resample pairs, average both deltas
            per_q = [(r[f"{kind}_delta_A_vs_{c}"] + r[f"{kind}_delta_B_vs_{c}"]) / 2 for r in rows]
            summ[f"{kind}_vs_{c}"] = {"mean": float(np.mean(v)), "ci95_by_question": boot(per_q), "frac_pos": float(np.mean([x > 0 for x in v])),
                                      "both_pos": sum(r[f"{kind}_delta_A_vs_{c}"] > 0 and r[f"{kind}_delta_B_vs_{c}"] > 0 for r in rows)}
        for c in ("correct", "swapped", "crossq", "mean", "zero"):
            summ[f"ce_{kind}_{c}"] = -float(np.mean([r[f"{kind}_s_A_{c}"] for r in rows] + [r[f"{kind}_s_B_{c}"] for r in rows]))
    return summ, rows


@torch.no_grad()
def eval_singles(E, singles, perm):
    out = {c: [] for c in ("correct", "crossq", "mean", "zero")}
    for i, s in enumerate(singles):
        z = E.Z[s["plan_hash"]]
        zs = torch.stack([z, E.Z[singles[perm[i]]["plan_hash"]], E.Zmean, torch.zeros_like(z)])
        for k, c in enumerate(out):
            sf, _, _ = E.scores(s["question"], zs[k:k + 1], [s["execution"]])
            out[c].append(-float(sf[0]))
    summ = {f"ce_{c}": float(np.mean(v)) for c, v in out.items()}
    for c in ("crossq", "mean", "zero"):
        d = [b - a for a, b in zip(out["correct"], out[c])]
        summ[f"gain_{c}"] = {"mean": float(np.mean(d)), "ci95": boot(d), "frac_pos": float(np.mean([x > 0 for x in d]))}
    return summ


def derange(n, seed):
    rng = random.Random(seed)
    while True:
        p = list(range(n)); rng.shuffle(p)
        if all(p[i] != i for i in range(n)):
            return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gates", "train", "eval"], required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50); ap.add_argument("--singles_per_rank", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--ckpt", default=None); ap.add_argument("--gen", action="store_true"); ap.add_argument("--gen_tokens", type=int, default=2048)
    a = ap.parse_args()
    ddp = a.mode == "train" and "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    os.makedirs(a.out, exist_ok=True)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    singles = load_jsonl(f"{a.data}/train_singles.jsonl"); val = load_jsonl(f"{a.data}/val_singles.jsonl")
    pairs = load_jsonl(f"{a.data}/pairs.jsonl")
    tpairs = [p for p in pairs if p["role"] == "train_pair"]; hpairs = [p for p in pairs if p["role"] == "heldout8"]
    E = Exec2B(a.model, a.data)
    params = E.params()

    if a.mode == "gates":
        from later.scripts.stage2a.plan_ae import key
        g = {"adapter_params": sum(p.numel() for p in E.adapter.parameters()), "lora_params": sum(p.numel() for _, p in E.lora),
             "lora_modules": sorted({n.split(".lora_")[0].split(".")[-1] for n, _ in E.lora}),
             "base_trainable": sum(p.numel() for n, p in E.peft.named_parameters() if p.requires_grad and "lora_" not in n)}
        base_snap = {n: p.detach().cpu().clone() for n, p in E.peft.named_parameters() if "lora_" not in n}
        p0 = hpairs[0]
        pre, slots = E.P.prefix_ids(p0["question"])
        dec = E.P.tok.decode(pre)
        g["G2_prefix_tail"] = E.P.tok.decode(pre[-14:])
        g["G2_no_plan_text_in_prefix"] = all(p0[k][20:100] not in dec for k in ("plan_A_norm", "plan_B_norm"))
        g["G2_no_execution_or_answer_text_in_prefix"] = p0["execution_A"][:80] not in dec and "</EXECUTION>" not in dec
        g["G3_z_shape"] = list(E.Z[key(p0["plan_A_norm"])].shape)
        g["G3_hidden_size"] = E.P.model.config.hidden_size
        t = pair_terms(E, p0)
        aux = t["aux"]
        with torch.autocast("cuda", dtype=torch.bfloat16):   # same precision path as E.scores
            emb_adapter = E.soft(torch.stack([E.Z[key(p0[k])] for k in ("plan_A_norm", "plan_B_norm", "plan_B_norm", "plan_A_norm")])).detach()
        g["G3_slots_equal_adapter_output_max_abs"] = float((aux["emb"][:, slots].float() - emb_adapter.to(torch.bfloat16).float()).abs().max())
        g["G3_positions_contiguous"] = bool((aux["pos"][0] == torch.arange(aux["pos"].size(1), device=E.dev)).all())
        L0 = len(pre)
        g["G3_prefix_ids_pos_mask_equal_4_seqs"] = bool((aux["ids"][:, :L0] == aux["ids"][0, :L0]).all() and (aux["att"][:, :L0] == 1).all())
        g["G3_logits_finite"] = bool(torch.isfinite(aux["first_logits"]).all())
        # first-token / shift: manual CE of the first execution token from the last prefix position
        with torch.no_grad():
            sf, ss, ax = E.scores(p0["question"], E.Z[key(p0["plan_A_norm"])].unsqueeze(0), [p0["execution_A"]])
            first_tok = E.P.target_ids(p0["execution_A"])[0]
            ids = ax["ids"]; emb = ax["emb"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = E.P.model.lm_head(E.P.model.model(inputs_embeds=emb, attention_mask=ax["att"], position_ids=ax["pos"], use_cache=False).last_hidden_state[0, L0 - 1]).float()
            g["G4_first_token_ce_manual_vs_scores"] = [float(F.cross_entropy(lg[None], torch.tensor([first_tok], device=E.dev))), float(-ax["first_logits"].log_softmax(-1)[first_tok])]
            g["G4_info_full_execution_ce_this_sample"] = float(-sf[0])
        g["G4_first_token_ok"] = abs(g["G4_first_token_ce_manual_vs_scores"][0] - g["G4_first_token_ce_manual_vs_scores"][1]) < 1e-3
        # counterfactual injection: swapping Z changes logits; zero/mean through the same adapter path
        with torch.no_grad():
            s1, _, a1 = E.scores(p0["question"], E.Z[key(p0["plan_A_norm"])].unsqueeze(0), [p0["execution_A"]])
            s2, _, a2 = E.scores(p0["question"], E.Z[key(p0["plan_B_norm"])].unsqueeze(0), [p0["execution_A"]])
            s3, _, a3 = E.scores(p0["question"], torch.zeros_like(E.Zmean).unsqueeze(0), [p0["execution_A"]])
        g["G5_swap_changes_slots_and_logits"] = float((a1["emb"] - a2["emb"]).abs().max()) > 0 and float((a1["first_logits"] - a2["first_logits"]).abs().max()) > 0
        g["G5_zero_through_adapter_finite"] = bool(torch.isfinite(a3["first_logits"]).all())
        g["G5_no_kv_cache_reuse"] = "use_cache=False in every scoring forward; generation builds a fresh prefix per call"
        # gradients
        loss = t["ce"][0] + t["ce"][1] + LAMBDA_RANK * t["rank"]
        loss.backward()
        ga = [p.grad for p in E.adapter.parameters()]; gl = [p.grad for n, p in E.lora if "lora_B" in n]
        g["G6_adapter_grad_finite_nonzero"] = all(x is not None and torch.isfinite(x).all().item() for x in ga) and float(sum(x.norm() for x in ga)) > 0
        g["G6_lora_B_grad_finite_nonzero"] = all(x is not None and torch.isfinite(x).all().item() for x in gl) and float(sum(x.norm() for x in gl)) > 0
        g["G6_base_grads_none"] = all(p.grad is None for n, p in E.peft.named_parameters() if "lora_" not in n)
        for p in params:
            p.grad = None
        # overfit smoke: 6 train singles, 5 steps
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
        sm = singles[:6]; hist = []
        for st in range(5):
            opt.zero_grad(set_to_none=True); tot = 0.0
            for s in sm:
                l, _ = single_loss(E, s); (l / len(sm)).backward(); tot += float(l) / len(sm)
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); hist.append(tot)
        g["G9_overfit_ce_by_step"] = hist; g["G9_pass"] = hist[-1] < hist[0]
        g["G6_optimizer_only_adapter_lora"] = {id(p) for gr in opt.param_groups for p in gr["params"]} == {id(p) for p in params}
        g["G6_base_bitwise_unchanged"] = all(torch.equal(p.detach().cpu(), base_snap[n]) for n, p in E.peft.named_parameters() if "lora_" not in n)
        g["G1_stage2a_Z_verified"] = json.load(open(f"{a.data}/stage2a_latent_cache_manifest.json"))["verify"]["abs_diff"] == 0.0
        # save / reload
        path = f"{a.out}/tmp_gate.pt"
        torch.save({"state": E.state(), "opt": opt.state_dict()}, path)
        E.adapter.eval(); E.peft.eval()
        with torch.no_grad():
            l1, _ = single_loss(E, sm[0])
        ck = torch.load(path, weights_only=False)
        with torch.no_grad():
            for p in params:
                p.add_(0.01)
        E.load(ck["state"])
        with torch.no_grad():
            l2, _ = single_loss(E, sm[0])
        g["G8_reload_loss"] = [float(l1), float(l2)]; g["G8_pass"] = abs(float(l1) - float(l2)) < 1e-5
        os.remove(path)
        # train path vs generation path
        with torch.no_grad():
            z = E.Z[key(p0["plan_A_norm"])]
            _, _, ax = E.scores(p0["question"], z.unsqueeze(0), [p0["execution_A"]])
            e = E.soft(z.unsqueeze(0))[0]
            pre_ids = torch.tensor([pre], device=E.dev)
            emb = E.P.model.get_input_embeddings()(pre_ids).clone(); emb[0, slots] = e.to(emb.dtype)
            gg = E.P.model.generate(inputs_embeds=emb, attention_mask=torch.ones_like(pre_ids), max_new_tokens=1, do_sample=False,
                                    output_scores=True, return_dict_in_generate=True, pad_token_id=E.P.T.pad_id)
            g["G5_train_vs_generate_first_argmax_equal"] = int(ax["first_logits"].argmax()) == int(gg.scores[0][0].argmax())
        hard = [g["base_trainable"] == 0, g["G1_stage2a_Z_verified"], g["G2_no_plan_text_in_prefix"], g["G2_no_execution_or_answer_text_in_prefix"],
                g["G3_z_shape"] == [8, g["G3_hidden_size"]], g["G3_slots_equal_adapter_output_max_abs"] == 0.0, g["G3_positions_contiguous"],
                g["G3_prefix_ids_pos_mask_equal_4_seqs"], g["G3_logits_finite"], g["G4_first_token_ok"], g["G5_swap_changes_slots_and_logits"],
                g["G5_zero_through_adapter_finite"], g["G5_train_vs_generate_first_argmax_equal"], g["G6_adapter_grad_finite_nonzero"],
                g["G6_lora_B_grad_finite_nonzero"], g["G6_base_grads_none"], g["G6_optimizer_only_adapter_lora"], g["G6_base_bitwise_unchanged"],
                g["G8_pass"], g["G9_pass"]]
        g["ALL_HARD_GATES_PASS"] = all(hard)
        json.dump(g, open(f"{a.out}/sanity_gates_single_gpu.json", "w"), indent=1)
        print(json.dumps(g, indent=1))
        return

    if a.mode == "train":
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
        rng = random.Random(42)
        os.makedirs(f"{a.out}/checkpoints", exist_ok=True)
        log = open(f"{a.out}/train_log.jsonl", "w") if rank == 0 else None
        base_fp = float(sum(p.detach().double().sum() for n, p in E.peft.named_parameters() if "lora_" not in n))
        def save(step):
            if rank == 0:
                torch.save({"state": E.state(), "opt": opt.state_dict(), "step": step, "rng_py": rng.getstate()}, f"{a.out}/checkpoints/step{step}.pt")
        save(0)
        s_order, p_order = [], []
        t0 = time.time(); torch.cuda.reset_peak_memory_stats()
        E.adapter.train(); E.peft.train()
        for step in range(1, a.steps + 1):
            if len(s_order) < a.singles_per_rank * world:
                idx = list(range(len(singles))); rng.shuffle(idx); s_order = idx + s_order
            if len(p_order) < world:
                idx = list(range(len(tpairs))); rng.shuffle(idx); p_order = idx + p_order
            gs = [s_order.pop() for _ in range(a.singles_per_rank * world)]
            gp = [p_order.pop() for _ in range(world)]
            swaps = [rng.random() < 0.5 for _ in range(world)]
            my_s, my_p, my_sw = gs[rank::world], gp[rank], swaps[rank]
            n_terms = len(my_s) + 2
            opt.zero_grad(set_to_none=True)
            ce_sum = 0.0
            for i in my_s:
                l, _ = single_loss(E, singles[i]); (l / n_terms).backward(); ce_sum += float(l)
            t = pair_terms(E, tpairs[my_p], swap=my_sw)
            ((t["ce"][0] + t["ce"][1]) / n_terms + LAMBDA_RANK * t["rank"]).backward()
            ce_sum += float(t["ce"][0] + t["ce"][1])
            local_gn = float(torch.sqrt(sum(p.grad.float().pow(2).sum() for p in params if p.grad is not None)))
            stats = torch.tensor([ce_sum / n_terms, float(t["rank"]), sum(t["margin_span"]) / 2, sum(t["margin_full"]) / 2, local_gn], device=E.dev)
            if ddp:
                for p in params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM); p.grad.div_(world)
                all_stats = [torch.zeros_like(stats) for _ in range(world)]; dist.all_gather(all_stats, stats)
            else:
                all_stats = [stats]
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            if ddp and step in (1, a.steps):
                fp = torch.tensor([float(sum(p.detach().double().sum() for p in params))], device=E.dev, dtype=torch.float64)
                fps = [torch.zeros_like(fp) for _ in range(world)]; dist.all_gather(fps, fp)
                params_identical = len({float(x) for x in fps}) == 1
            else:
                params_identical = None
            if rank == 0:
                S_ = torch.stack(all_stats).cpu().numpy()
                rec = {"step": step, "train_ce": float(S_[:, 0].mean()), "rank_loss": float(S_[:, 1].mean()), "pair_margin_span": float(S_[:, 2].mean()),
                       "pair_margin_full": float(S_[:, 3].mean()), "grad_norm_global": gn, "grad_norm_local_per_rank": S_[:, 4].tolist(),
                       "singles_global": [singles[i]["id"] for i in gs], "pairs_global": [tpairs[i]["id"] for i in gp], "world": world,
                       "params_identical_across_ranks": params_identical, "elapsed_s": time.time() - t0,
                       "peak_mem_gb_rank0": torch.cuda.max_memory_allocated() / 2**30}
                log.write(json.dumps(rec) + "\n"); log.flush()
                if step <= 10 or step % 5 == 0:
                    print("step %d ce %.4f rank %.4f margin %.4f gn %.3f local %s %.1fs/step mem %.1fGB" % (step, rec["train_ce"], rec["rank_loss"],
                          rec["pair_margin_span"], gn, [round(x, 3) for x in rec["grad_norm_local_per_rank"]], rec["elapsed_s"] / step, rec["peak_mem_gb_rank0"]), flush=True)
            if step in (25, 50):
                save(step)
        mem = torch.tensor([torch.cuda.max_memory_allocated() / 2**30], device=E.dev)
        mems = [torch.zeros_like(mem) for _ in range(world)]
        if ddp:
            dist.all_gather(mems, mem)
        else:
            mems = [mem]
        base_fp2 = float(sum(p.detach().double().sum() for n, p in E.peft.named_parameters() if "lora_" not in n))
        if rank == 0:
            json.dump({"train_time_s": time.time() - t0, "world": world, "peak_mem_gb_per_rank": [float(m) for m in mems],
                       "base_param_sum_unchanged": base_fp == base_fp2, "global_batch": f"{a.singles_per_rank * world} singles + {world} pairs per step"},
                      open(f"{a.out}/post_training_checks.json", "w"), indent=1)
        if ddp:
            dist.barrier(); dist.destroy_process_group()
        return

    if a.mode == "eval":
        ck = torch.load(a.ckpt, weights_only=False); E.load(ck["state"]); step = ck["step"]
        E.adapter.eval(); E.peft.eval()
        cross = derange(len(hpairs), 42)
        hs, hrows = eval_pairs(E, hpairs, cross)
        ts, _ = eval_pairs(E, tpairs, derange(len(tpairs), 42))
        vs = eval_singles(E, val, derange(len(val), 42))
        train_ce = float(np.mean([float(single_loss(E, s)[0]) for s in singles[:48]]))
        res = {"step": step, "heldout_pairs": hs, "heldout_rows": hrows, "train_pairs": ts, "val_singles": vs, "train_ce_48": train_ce,
               "adapter_scale": float(E.adapter.log_scale.exp())}
        if a.gen:
            from later.scripts.stage2a.plan_ae import key
            gens = []
            for p in hpairs:
                for c, k in (("Z_A", "plan_A_norm"), ("Z_B", "plan_B_norm")):
                    g = E.generate(p["question"], E.Z[key(p[k])], a.gen_tokens)
                    pred = last_boxed(g)
                    gens.append({"id": p["id"], "condition": c, "generation": g, "pred": pred, "correct": correct(pred, str(p["answer"])),
                                 "overlap_E_A": ngram_overlap(g, p["execution_A"]), "overlap_E_B": ngram_overlap(g, p["execution_B"]),
                                 "method_terms_A_hits": method_hits(g, p["method_terms_only_A"]), "method_terms_B_hits": method_hits(g, p["method_terms_only_B"]),
                                 "ended": g.rstrip().endswith("<|im_end|>")})
            res["gen_summary"] = {"correct": sum(x["correct"] for x in gens), "ended": sum(x["ended"] for x in gens),
                                  "identical_pairs": sum(gens[2 * i]["generation"] == gens[2 * i + 1]["generation"] for i in range(len(hpairs)))}
            with open(f"{a.out}/evaluations/generations_step{step}.jsonl", "w") as f:
                for x in gens:
                    f.write(json.dumps(x) + "\n")
        os.makedirs(f"{a.out}/evaluations", exist_ok=True)
        json.dump(res, open(f"{a.out}/evaluations/heldout_step{step}.json", "w"), indent=1)
        print("EVAL", step, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if not kk.startswith("ce_span")})
                                        for k, v in res.items() if k not in ("heldout_rows",)})[:3000])


if __name__ == "__main__":
    main()
