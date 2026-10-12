"""Stage 2 three-step pilot: frozen 2A Plan Encoder -> 2B autonomous Latent-KD warm-up -> 2C (KD + Exec [+ Method Choice]).

Student = ONE Qwen3-4B (Stage-1 step 800) + ONE shared LoRA (r16 q/v/o) + the LaTER LatentProjector (fresh init, output
per-vector RMS = 4B token-embedding RMS). Deterministic recurrence (sigma = 0): chat(Q)+'<EXPLORATION>\\n' -> z1..z8 ->
'</EXPLORATION><EXECUTION>\\n' -> E + '</EXECUTION><|im_end|>'.  (Student class reused from stage2s; noise always 0.)
Teacher (train only) = raw Stage-2A step-300 compressor output Z_T = C(P) [8, 2560] (cached in the 2B z_cache). The 2A
compressor already RMS-calibrates every vector to the 4B token-embedding RMS, and the 2A reader consumes it as
`z / tok_rms`, i.e. Z_T is defined for the same 8 input-embedding slots that Z_S is written into -> no adapter (the 2B
step-50 adapter is NOT used: it was trained on the train pairs' executions).
L_KD   = mean((Z_S - sg(Z_T))^2) / tok_rms^2        (plain MSE, constant rescale so that Z_S = 0 gives 1)
L_Exec = - mean log p(E,A | Q, Z_S)                    (Z_S autonomous, gradient through recurrence + projector)
L_Choice = softplus(d - m(Q,Z_TA)) + softplus(d + m(Q,Z_TB)) + eta * softplus(d - sgn * m(Q,Z_S)),
           m = s_A - s_B, s_X = mean log p(D_X | Q, Z, C) on the audited method-decision span (C = token LCP of E_A/E_B).
Modes: gates | train (--phase 2b|2c [--choice]) | audit (choice gradient audit) | eval.  All single-process / single-GPU;
the two 2C branches run concurrently, one per GPU, same data order.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from later.scripts.bridge.bridge_input_abcd import boot
from later.scripts.bridge.mini_bridge import correct, last_boxed, method_hits, ngram_overlap
from later.scripts.stage2s.stochastic_pilot import CHOICE_VALID, EXCLUDED, L, Student, build_pairs, grad_norm, key, short

TOK_RMS = 0.02195320464670658
D = 2560
TINY_SINGLES = 4


class TStudent(Student):
    @torch.no_grad()
    def teacher(self, plan):                              # raw 2A Z_T, no adapter
        return self.Zraw[key(plan)].to(self.dev).float()

    def zlat(self, q):
        return self.latents(q, torch.zeros(1, D, device=self.dev), 0.0, 1.0)[0]

    def score_nolat(self, q, targets, spans):
        """Same layout without the 8 latent slots (positions contiguous)."""
        pre = self.prompt_ids(q) + self.T.suffix_ids
        K, n = len(targets), len(pre) + max(len(t) for t in targets)
        ids = torch.full((K, n), self.T.pad_id, dtype=torch.long); lab = torch.full((K, n), -100, dtype=torch.long)
        att = torch.zeros(K, n, dtype=torch.long)
        for i, t in enumerate(targets):
            ids[i, :len(pre) + len(t)] = torch.tensor(pre + t); lab[i, len(pre):len(pre) + len(t)] = torch.tensor(t); att[i, :len(pre) + len(t)] = 1
        ids, lab, att = ids.to(self.dev), lab.to(self.dev), att.to(self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = self.base.model(input_ids=ids, attention_mask=att, position_ids=torch.arange(n, device=self.dev).unsqueeze(0).expand(K, -1), use_cache=False).last_hidden_state
        out_f, out_s = [], []
        for i, t in enumerate(targets):
            lg = self.base.lm_head(hs[i, len(pre) - 1:len(pre) + len(t) - 1]).float()
            lp = -F.cross_entropy(lg, torch.tensor(t, device=self.dev), reduction="none")
            out_f.append(lp.mean()); a, b = spans[i]; out_s.append(lp[a:b].mean())
        return torch.stack(out_f), torch.stack(out_s)


def kd(Z, T):
    return ((Z - T) ** 2).mean() / TOK_RMS ** 2


def default_is_A(pid):
    return int(key(pid), 16) % 2 == 0


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def single_loss(S, s, lam):
    Z = S.zlat(s["question"]); T = S.teacher(s["plan_norm"])
    l_kd = kd(Z, T)
    if lam["exec"] > 0:
        sf, _, _ = S.score(s["question"], Z.unsqueeze(0), [S.target_ids(s["execution"])])
        l_exec = -sf[0]
    else:
        l_exec = torch.zeros((), device=S.dev)
    return lam["kd"] * l_kd + lam["exec"] * l_exec, {"kd": l_kd, "exec": l_exec}


def choice_terms(S, p, Z, delta, sgn):
    TA, TB = S.teacher(p["plan_A_norm"]), S.teacher(p["plan_B_norm"])
    C, DA, DB = p["C"], p["D_A"], p["D_B"]
    sp = lambda X: (len(C), len(C) + len(X))
    _, ss, _ = S.score(p["question"], torch.stack([TA, TA, TB, TB, Z, Z]), [C + DA, C + DB] * 3, [sp(DA), sp(DB)] * 3)
    m_TA, m_TB, m_S = ss[0] - ss[1], ss[2] - ss[3], ss[4] - ss[5]
    ext = F.softplus(delta - m_TA) + F.softplus(delta + m_TB)
    slf = F.softplus(delta - sgn * m_S)
    return ext, slf, {"m_TA": float(m_TA), "m_TB": float(m_TB), "m_S": float(m_S)}


def pair_loss(S, p, lam, use_choice, delta, eta):
    """default plan (fixed per question) -> KD + Exec ; + Choice if use_choice and the pair is choice-valid."""
    dA = default_is_A(p["id"]); sgn = 1.0 if dA else -1.0
    plan, tgt = (p["plan_A_norm"], p["tgt_A"]) if dA else (p["plan_B_norm"], p["tgt_B"])
    Z = S.zlat(p["question"])
    l_kd = kd(Z, S.teacher(plan))
    sf, _, _ = S.score(p["question"], Z.unsqueeze(0), [tgt])
    l_exec = -sf[0]
    main = lam["kd"] * l_kd + lam["exec"] * l_exec
    parts = {"kd": l_kd, "exec": l_exec}
    if use_choice and p["choice_valid"]:
        ext, slf, info = choice_terms(S, p, Z, delta, sgn)
        parts.update({"choice_ext": ext, "choice_self": slf, "choice": ext + eta * slf}); parts.update(info)
    return main, parts


def data(a, S):
    singles = load_jsonl(f"{a.s2b}/data/train_singles.jsonl")
    vals = load_jsonl(f"{a.s2b}/data/val_singles.jsonl")
    pairs = build_pairs(S, a.s2b)
    trainp = [p for p in pairs if p["role"] == "train_pair" and not p["excluded"]]
    choicep = [p for p in trainp if p["choice_valid"]]
    held = [p for p in pairs if p["role"] == "heldout8"]
    return singles, vals, pairs, trainp, choicep, held


def batches(singles, choicep, n_single, n_pair, steps, seed):
    """Deterministic data order shared by both 2C branches."""
    rng = random.Random(seed); out = []; so, po = [], []
    for _ in range(steps):
        if len(so) < n_single:
            i = list(range(len(singles))); rng.shuffle(i); so += i
        if len(po) < n_pair:
            i = list(range(len(choicep))); rng.shuffle(i); po += i
        out.append(([so.pop(0) for _ in range(n_single)], [po.pop(0) for _ in range(n_pair)] if n_pair else []))
    return out


def step_loss(S, B, singles, choicep, lam, use_choice, delta, eta, lam_choice):
    si, pi = B
    n = len(si) + len(pi); recs = []
    for i in si:
        loss, parts = single_loss(S, singles[i], lam)
        (loss / n).backward(); recs.append({k: float(v) for k, v in parts.items()} | {"kind": "single"})
    for i in pi:
        main, parts = pair_loss(S, choicep[i], lam, use_choice, delta, eta)
        tot = main / n
        if "choice" in parts:
            tot = tot + lam_choice * parts["choice"] / len(pi)
        tot.backward(); recs.append({k: float(v) for k, v in parts.items()} | {"kind": "pair", "id": short(choicep[i]["id"])})
    return recs


def save_ck(S, opt, path, step, extra):
    torch.save({"state": S.state(), "opt": opt.state_dict() if opt else None, "step": step, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(), **extra}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gates", "train", "audit", "eval"], required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--s2b", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--phase", default="2b"); ap.add_argument("--choice", action="store_true"); ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--init", default=None); ap.add_argument("--steps", type=int, default=40); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_single", type=int, default=16); ap.add_argument("--n_pair", type=int, default=0)
    ap.add_argument("--lam_kd", type=float, default=1.0); ap.add_argument("--lam_exec", type=float, default=1.0)
    ap.add_argument("--lam_choice", type=float, default=1.0); ap.add_argument("--delta", type=float, default=0.0); ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--tag", required=False, default="run"); ap.add_argument("--eval_at", default="")
    ap.add_argument("--ckpt", default=None); ap.add_argument("--gen", action="store_true"); ap.add_argument("--gen_tokens", type=int, default=2048)
    ap.add_argument("--n_val", type=int, default=48); ap.add_argument("--dec", default="Qwen/Qwen3-0.6B"); ap.add_argument("--a2", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(42); np.random.seed(42)
    S = TStudent(a.model, a.s2b, seed=0)
    singles, vals, pairs, trainp, choicep, held = data(a, S)
    params = S.params()
    lam = {"kd": a.lam_kd, "exec": a.lam_exec}

    if a.mode == "gates":
        g = {}
        p0 = next(p for p in choicep if short(p["id"]) == "s1val-00645"); q = p0["question"]; s0 = singles[0]
        # 1 shared params
        g["G1_trainable"] = {"projector": sum(p.numel() for _, p in S.proj), "lora": sum(p.numel() for _, p in S.lora),
                             "lora_modules": sorted({n.split(".lora_")[0].split(".")[-1] for n, _ in S.lora}),
                             "n_adapters": len(S.peft.peft_config), "adapter_names": list(S.peft.peft_config)}
        g["G1_single_adapter"] = len(S.peft.peft_config) == 1
        # 2 teacher-only plan / no leak into student prompt
        dec = S.tok.decode(S.prompt_ids(q)); decs = S.tok.decode(S.prompt_ids(s0["question"]))
        leak = any(t[10:80] in dec for t in (p0["plan_A_norm"], p0["plan_B_norm"], p0["execution_A"], p0["execution_B"])) or \
            any(t[10:80] in decs for t in (s0["plan_norm"], s0["execution"])) or any(lb and lb in dec for lb in (p0["method_label_A"], p0["method_label_B"]))
        g["G2_no_plan_exec_label_in_student_prompt"] = not leak
        g["G2_teacher_no_grad"] = not S.teacher(s0["plan_norm"]).requires_grad
        g["G2_prompt_tail"] = S.tok.decode(S.prompt_ids(q)[-10:]) + " [8 slots] " + S.tok.decode(S.T.suffix_ids)
        # 3 latent autonomous: perturbing the teacher cache does not change Z_S
        with torch.no_grad():
            z1 = S.zlat(q); k = key(p0["plan_A_norm"]); bak = S.Zraw[k].clone(); S.Zraw[k] = S.Zraw[k] * 0 + 1.0
            z2 = S.zlat(q); S.Zraw[k] = bak
        g["G3_zs_independent_of_teacher_cache"] = bool(torch.equal(z1, z2))
        g["G3_zs_deterministic"] = bool(torch.equal(z1, S.zlat(q).detach()))
        # 4 injection: each slot changes the first execution logits; train vs generate first token
        with torch.no_grad():
            _, _, a0 = S.score(q, z1[None], [p0["tgt_A"]])
            ch = []
            for t in range(L):
                z = z1.clone(); z[t] = 0
                _, _, at = S.score(q, z[None], [p0["tgt_A"]]); ch.append(float((at["first_logits"] - a0["first_logits"]).abs().max()))
            _, first = S.generate(q, z1[None], max_new=1)
        g["G4_slot_effect_on_first_logits"] = ch; g["G4_all_slots_visible"] = all(x > 0 for x in ch)
        g["G4_train_vs_generate_first_argmax_equal"] = int(a0["first_logits"].argmax()) == int(first[0].argmax())
        # 5 gradient paths
        gr = {}
        def bw(name, fn):
            for p_ in params:
                p_.grad = None
            Z = S.latents(q, torch.zeros(1, D, device=S.dev), 0.0, 1.0)
            Z.retain_grad()
            fn(Z[0]).backward()
            gr[name] = {"projector": grad_norm([p for _, p in S.proj]), "projector_first_layer": grad_norm([p for n, p in S.proj if n.startswith("net.0")]),
                        "lora": grad_norm([p for _, p in S.lora]), "dZ_S": float(Z.grad.norm()) if Z.grad is not None else 0.0,
                        "dz1": float(Z.grad[0, 0].norm()) if Z.grad is not None else 0.0,
                        "finite": all(torch.isfinite(p.grad).all().item() for p in params if p.grad is not None)}
        bw("kd", lambda Z: kd(Z, S.teacher(p0["plan_A_norm"])))
        bw("exec", lambda Z: -S.score(q, Z[None], [p0["tgt_A"]])[0][0])
        bw("choice_self", lambda Z: choice_terms(S, p0, Z, 0.0, 1.0)[1])
        bw("choice_ext", lambda Z: choice_terms(S, p0, Z, 0.0, 1.0)[0] + 0 * Z.sum())
        g["G5_grads"] = gr
        g["G5_ok"] = all(gr[n]["projector"] > 0 and gr[n]["lora"] > 0 and gr[n]["dz1"] > 0 and gr[n]["finite"] for n in ("kd", "exec", "choice_self")) and \
            gr["choice_ext"]["projector"] == 0 and gr["choice_ext"]["lora"] > 0
        for p_ in params:
            p_.grad = None
        # 6 masks / spans
        with torch.no_grad():
            C, DA = p0["C"], p0["D_A"]
            _, ss1, _ = S.score(q, z1[None], [C + DA], [(len(C), len(C) + len(DA))])
            _, ss2, _ = S.score(q, z1[None], [p0["tgt_A"]], [(len(C), len(C) + len(DA))])
        t = p0["tgt_A"]
        g["G6_target_ends"] = S.tok.decode(t[-6:]); g["G6_target_ends_with_eos"] = t[-1] == S.tok.convert_tokens_to_ids("<|im_end|>")
        g["G6_span_short_vs_long"] = [float(ss1[0]), float(ss2[0])]; g["G6_span_consistent"] = abs(float(ss1[0]) - float(ss2[0])) < 0.05
        g["G6_C_text"] = p0["C_text"]; g["G6_D_text"] = [p0["D_A_text"], p0["D_B_text"]]
        # 7 four-grid sanity: identical Z for both rows -> m_TA == m_TB, cannot be "both correct"
        with torch.no_grad():
            pp = dict(p0); pp["plan_B_norm"] = p0["plan_A_norm"]
            _, _, inf_same = choice_terms(S, pp, z1, 0.0, 1.0); _, _, inf = choice_terms(S, p0, z1, 0.0, 1.0)
        g["G7_grid_same_Z"] = inf_same; g["G7_grid_real"] = inf
        g["G7_same_Z_not_both_correct"] = not (inf_same["m_TA"] > 0 and inf_same["m_TB"] < 0) and abs(inf_same["m_TA"] - inf_same["m_TB"]) < 1e-4
        # 8 interventions finite
        with torch.no_grad():
            Zx = S.zlat(trainp[1]["question"]); Zm = torch.stack([S.zlat(s["question"]) for s in singles[:8]]).mean(0)
            res = {}
            for nm, Z in (("self", z1), ("zero", torch.zeros_like(z1)), ("mean", Zm), ("crossq", Zx), ("teacher_B", S.teacher(p0["plan_B_norm"]))):
                sf, ss, _ = S.score(q, Z[None], [C + DA], [(len(C), len(C) + len(DA))]); res[nm] = float(ss[0])
            sf, ss = S.score_nolat(q, [C + DA], [(len(C), len(C) + len(DA))]); res["nolatent"] = float(ss[0])
        g["G8_span_scores"] = res; g["G8_finite"] = all(math.isfinite(v) for v in res.values())
        # 10 reload
        snap = {n: p.detach().clone() for n, p in S.peft.named_parameters() if not p.requires_grad}
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        for p_ in params:
            p_.grad = torch.randn_like(p_) * 1e-3
        opt.step(); opt.zero_grad(set_to_none=True)
        path = f"{a.out}/tmp_gate.pt"; save_ck(S, opt, path, 1, {})
        with torch.no_grad():
            l1 = float(pair_loss(S, p0, lam, True, 0.0, 1.0)[1]["choice"])
        st0 = torch.load(path, weights_only=False)
        S2 = S.state()
        with torch.no_grad():
            for n, p in S.proj:
                p.add_(1.0)
        S.load(st0["state"]); opt.load_state_dict(st0["opt"])
        with torch.no_grad():
            l2 = float(pair_loss(S, p0, lam, True, 0.0, 1.0)[1]["choice"])
        g["G10_reload"] = [l1, l2]; g["G10_reload_ok"] = abs(l1 - l2) < 1e-5
        g["G10_base_unchanged"] = all(torch.equal(p, snap[n]) for n, p in S.peft.named_parameters() if not p.requires_grad)
        os.remove(path)
        g["G0_teacher_rms"] = float(S.teacher(s0["plan_norm"]).pow(2).mean().sqrt()); g["G0_tok_rms"] = S.tok_rms
        g["G0_student_rms_init"] = float(z1.pow(2).mean().sqrt())
        g["G0_kd_init_self"] = float(kd(z1, S.teacher(p0["plan_A_norm"])))
        hard = [g["G1_single_adapter"], g["G2_no_plan_exec_label_in_student_prompt"], g["G2_teacher_no_grad"], g["G3_zs_independent_of_teacher_cache"],
                g["G3_zs_deterministic"], g["G4_all_slots_visible"], g["G4_train_vs_generate_first_argmax_equal"], g["G5_ok"], g["G6_target_ends_with_eos"],
                g["G6_span_consistent"], g["G7_same_Z_not_both_correct"], g["G8_finite"], g["G10_reload_ok"], g["G10_base_unchanged"]]
        g["hard_gate_vector"] = hard; g["ALL_HARD_GATES_PASS"] = all(hard)
        json.dump(g, open(f"{a.out}/gates.json", "w"), indent=1)
        print(json.dumps(g, indent=1))
        return

    if a.mode == "train":
        if a.init:
            S.load(torch.load(a.init, weights_only=False)["state"])
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        if a.tiny:            # tiny overfit: 4 singles + 2 choice pairs, all losses, every step the same items
            B = [(list(range(TINY_SINGLES)), [0, 1])] * a.steps
        else:
            B = batches(singles, choicep, a.n_single, a.n_pair, a.steps, seed=42 if a.phase == "2c" else 7)
        use_choice = a.choice
        log = open(f"{a.out}/train_{a.tag}.jsonl", "w")
        ckdir = f"{a.out}/checkpoints"; os.makedirs(ckdir, exist_ok=True)
        eval_at = {int(x) for x in a.eval_at.split(",") if x}
        if not a.tiny:
            save_ck(S, None, f"{ckdir}/{a.tag}_step0.pt", 0, {"phase": a.phase})
        def tiny_probe():
            with torch.no_grad():
                Zq = [S.zlat(singles[i]["question"]) for i in range(TINY_SINGLES)]; Tq = [S.teacher(singles[i]["plan_norm"]) for i in range(TINY_SINGLES)]
                return {"kd_correct": [float(kd(Zq[i], Tq[i])) for i in range(TINY_SINGLES)],
                        "kd_shuffled": [float(kd(Zq[i], Tq[(i + 1) % TINY_SINGLES])) for i in range(TINY_SINGLES)],
                        "choice": [choice_terms(S, choicep[j], S.zlat(choicep[j]["question"]), a.delta, 1.0 if default_is_A(choicep[j]["id"]) else -1.0)[2] for j in (0, 1)]}
        probe0 = tiny_probe() if a.tiny else None
        t0 = time.time(); torch.cuda.reset_peak_memory_stats()
        for step in range(1, a.steps + 1):
            opt.zero_grad(set_to_none=True)
            recs = step_loss(S, B[step - 1], singles, choicep, lam, use_choice, a.delta, a.eta, a.lam_choice)
            gproj = grad_norm([p for _, p in S.proj]); glora = grad_norm([p for _, p in S.lora])
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            mean = lambda k, kind=None: float(np.mean([r[k] for r in recs if k in r and (kind is None or r["kind"] == kind)])) if any(k in r for r in recs) else None
            rec = {"step": step, "kd": mean("kd"), "exec": mean("exec") if a.lam_exec > 0 else None, "choice": mean("choice"), "choice_ext": mean("choice_ext"),
                   "choice_self": mean("choice_self"), "m_TA": mean("m_TA"), "m_TB": mean("m_TB"), "m_S": mean("m_S"),
                   "grad_total_preclip": gn, "grad_proj": gproj, "grad_lora": glora, "elapsed_s": time.time() - t0,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "batch": B[step - 1], "per_item": recs}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"[{a.tag}] step {step} kd {rec['kd']:.4f} exec {rec['exec']} choice {rec['choice']} mTA {rec['m_TA']} mTB {rec['m_TB']} mS {rec['m_S']} "
                  f"gn {gn:.3f} (p {gproj:.3f} l {glora:.3f}) {rec['elapsed_s']:.0f}s mem {rec['peak_mem_gb']:.0f}G", flush=True)
            if step in eval_at or (step == a.steps and not a.tiny):
                save_ck(S, opt, f"{ckdir}/{a.tag}_step{step}.pt", step, {"phase": a.phase, "lam": lam, "lam_choice": a.lam_choice if use_choice else 0.0,
                                                                        "delta": a.delta, "eta": a.eta, "lr": a.lr})
        if a.tiny:            # tiny overfit report: before/after on the same items (+ shuffled-target KD)
            res = {"before": probe0, "after": tiny_probe(), "items": [singles[i]["id"] for i in range(TINY_SINGLES)] + [choicep[j]["id"] for j in (0, 1)],
                   "defaults": [default_is_A(choicep[j]["id"]) for j in (0, 1)]}
            json.dump(res, open(f"{a.out}/tiny_overfit.json", "w"), indent=1); print(json.dumps(res, indent=1))
        return

    if a.mode == "audit":
        S.load(torch.load(a.ckpt, weights_only=False)["state"])
        B = batches(singles, choicep, a.n_single, a.n_pair, 4, seed=42)       # = the first 4 2C batches
        rows = []
        def gvec():
            return torch.cat([p.grad.flatten().float() if p.grad is not None else torch.zeros(p.numel(), device=S.dev) for p in params])
        for bi, (si, pi) in enumerate(B):
            n = len(si) + len(pi)
            for p_ in params:
                p_.grad = None
            for i in si:
                loss, _ = single_loss(S, singles[i], lam); (loss / n).backward()
            for i in pi:
                main, _ = pair_loss(S, choicep[i], lam, False, a.delta, a.eta); (main / n).backward()
            g_main = gvec(); gm_proj = grad_norm([p for _, p in S.proj]); gm_lora = grad_norm([p for _, p in S.lora])
            terms = {}
            for nm in ("kd", "exec", "choice_ext", "choice_self"):
                for p_ in params:
                    p_.grad = None
                for i in pi:
                    _, parts = pair_loss(S, choicep[i], lam, True, a.delta, a.eta)
                    (parts[nm] / len(pi)).backward()
                terms[nm] = (gvec(), grad_norm([p for _, p in S.proj]), grad_norm([p for _, p in S.lora]))
            g_choice = terms["choice_ext"][0] + a.eta * terms["choice_self"][0]
            rows.append({"batch": bi, "pairs": [short(choicep[i]["id"]) for i in pi], "g_main_total": float(g_main.norm()), "g_main_proj": gm_proj, "g_main_lora": gm_lora,
                         **{f"g_pair_{nm}_total": float(v[0].norm()) for nm, v in terms.items()}, **{f"g_pair_{nm}_proj": v[1] for nm, v in terms.items()},
                         **{f"g_pair_{nm}_lora": v[2] for nm, v in terms.items()},
                         "g_choice_total(eta)": float(g_choice.norm()), "cos_main_choice": float(F.cosine_similarity(g_main, g_choice, 0))})
            print(rows[-1], flush=True)
        r = float(np.median([x["g_main_total"] for x in rows])) / float(np.median([x["g_choice_total(eta)"] for x in rows]))
        lam_c = 0.5 * r
        res = {"rule": "lambda_choice = 0.5 * median||grad L_main(step batch)|| / median||grad mean_pairs L_choice||, pre-registered, computed once at the 2B checkpoint on the first 4 2C train batches",
               "ratio": r, "lambda_choice": lam_c, "pair_fraction_per_step": f"{a.n_pair} choice-valid pairs / {a.n_single + a.n_pair} items", "rows": rows}
        json.dump(res, open(f"{a.out}/choice_gradient_audit.json", "w"), indent=1)
        with open(f"{a.out}/choice_gradient_audit.csv", "w") as f:
            ks = list(rows[0].keys()); f.write(",".join(ks) + "\n")
            for x in rows:
                f.write(",".join(json.dumps(x[k]).replace(",", ";") for k in ks) + "\n")
        print("LAMBDA_CHOICE", lam_c, "ratio", r)
        return

    if a.mode == "eval":
        ck = torch.load(a.ckpt, weights_only=False); S.load(ck["state"])
        name = os.path.basename(a.ckpt).replace(".pt", "")
        with torch.no_grad():
            Zm = torch.stack([S.zlat(s["question"]) for s in singles[:64]]).mean(0)
            Zm = Zm / Zm.pow(2).mean(-1, keepdim=True).sqrt() * S.tok_rms
            Tmean = torch.stack([S.teacher(s["plan_norm"]) for s in singles]).mean(0)
        reader = None
        if a.a2:
            from later.scripts.stage2a.plan_ae import System
            reader = System(a.dec, f"{a.a2}/data", seed=42)
            reader.load(torch.load(f"{a.a2}/checkpoints/step300.pt", weights_only=False)["state"]); reader.peft.eval()
        out = {"ckpt": a.ckpt, "step": ck.get("step"), "phase": ck.get("phase")}
        # ---- val singles (unseen questions): KD correct vs shuffled, latent geometry, exec CE interventions, 2A reader
        V = vals[:a.n_val]; vrows = []
        with torch.no_grad():
            ZS = torch.stack([S.zlat(v["question"]) for v in V]); ZT = torch.stack([S.teacher(v["plan_norm"]) for v in V])
            for i, v in enumerate(V):
                j = (i + 1) % len(V)
                tgt = S.target_ids(v["execution"])
                conds = {"self": ZS[i], "crossq": ZS[j], "mean": Zm, "zero": torch.zeros_like(Zm), "teacher": ZT[i]}
                ce = {}
                for nm, Z in conds.items():
                    sf, _, _ = S.score(v["question"], Z[None], [tgt]); ce[nm] = -float(sf[0])
                sf, _ = S.score_nolat(v["question"], [tgt], [(0, 1)]); ce["nolatent"] = -float(sf[0])
                r = {"id": v["id"], "kd_correct": float(kd(ZS[i], ZT[i])), "kd_shuffled": float(kd(ZS[i], ZT[j])), "kd_teacher_mean": float(kd(Tmean, ZT[i])),
                     "kd_student_mean": float(kd(Zm, ZT[i])), "exec_ce": ce}
                if reader is not None:
                    lps = reader.score(torch.stack([ZS[i], ZS[j], ZT[i], Zm, ZT[j]]), [v["plan_norm"]] * 5)
                    r["reader_ce"] = dict(zip(["self", "crossq", "teacher", "mean", "teacher_crossq"], [-float(x.mean()) for x in lps]))
                vrows.append(r)
            fl = ZS.flatten(1); cs = F.cosine_similarity(fl.unsqueeze(1), fl.unsqueeze(0), -1); iu = torch.triu_indices(len(V), len(V), 1)
            ft = ZT.flatten(1); ct = F.cosine_similarity(ft.unsqueeze(1), ft.unsqueeze(0), -1)
            geo = {"student_crossq_cos": float(cs[iu[0], iu[1]].mean()), "teacher_crossq_cos": float(ct[iu[0], iu[1]].mean()),
                   "student_var_over_teacher_var": float(ZS.var(0).mean() / ZT.var(0).mean()),
                   "student_var_per_slot": [float(x) for x in ZS.var(0).mean(-1) / ZT.var(0).mean(-1)],
                   "student_rms": float(ZS.pow(2).mean().sqrt()), "cos_student_teacher_matched": float(F.cosine_similarity(fl, ft, -1).mean())}
        def dstat(f):
            v = [f(r) for r in vrows]; return {"mean": float(np.mean(v)), "ci95": boot(v), "frac_pos": float(np.mean([x > 0 for x in v]))}
        vs = {"n": len(V), "kd_correct": float(np.mean([r["kd_correct"] for r in vrows])), "kd_shuffled": float(np.mean([r["kd_shuffled"] for r in vrows])),
              "kd_teacher_mean_baseline": float(np.mean([r["kd_teacher_mean"] for r in vrows])), "kd_student_mean": float(np.mean([r["kd_student_mean"] for r in vrows])),
              "kd_shuffled_minus_correct": dstat(lambda r: r["kd_shuffled"] - r["kd_correct"]), "geometry": geo}
        for c in ("crossq", "mean", "zero", "teacher", "nolatent"):
            vs[f"exec_ce_{c}_minus_self"] = dstat(lambda r, c=c: r["exec_ce"][c] - r["exec_ce"]["self"])
        vs["exec_ce_self"] = float(np.mean([r["exec_ce"]["self"] for r in vrows]))
        if reader is not None:
            for c in ("crossq", "teacher", "mean", "teacher_crossq"):
                vs[f"reader_ce_{c}_minus_self"] = dstat(lambda r, c=c: r["reader_ce"][c] - r["reader_ce"]["self"])
            vs["reader_ce_self"] = float(np.mean([r["reader_ce"]["self"] for r in vrows]))
        out["val_singles"] = vs
        # ---- pairs: 4-grid external, autonomous interventions, exec CE, free-run
        prow = []
        groups = {"heldout8": held, "train_pair": trainp}
        for p in held + trainp:
            grp = groups[p["role"]]; donor = grp[(grp.index(p) + 1) % len(grp)]
            q = p["question"]; dA = default_is_A(p["id"]); sgn = 1 if dA else -1
            with torch.no_grad():
                Zs = S.zlat(q); Zx = S.zlat(donor["question"]); TA, TB = S.teacher(p["plan_A_norm"]), S.teacher(p["plan_B_norm"])
                names = ["self", "crossq", "mean", "zero", "teacher_A", "teacher_B"]
                Zall = torch.stack([Zs, Zx, Zm, torch.zeros_like(Zm), TA, TB])
                C, DA, DB = p["C"], p["D_A"], p["D_B"]
                n = len(names)
                _, sA, _ = S.score(q, Zall, [C + DA] * n, [(len(C), len(C) + len(DA))] * n)
                _, sB, _ = S.score(q, Zall, [C + DB] * n, [(len(C), len(C) + len(DB))] * n)
                _, nl = S.score_nolat(q, [C + DA, C + DB], [(len(C), len(C) + len(DA)), (len(C), len(C) + len(DB))])
                pref = dict(zip(names, (sA - sB).tolist())); pref["nolatent"] = float(nl[0] - nl[1])
                tdef = p["tgt_A"] if dA else p["tgt_B"]
                ce = {}
                for nm, Z in zip(names[:4], Zall[:4]):
                    sf, _, _ = S.score(q, Z[None], [tdef]); ce[nm] = -float(sf[0])
                sf, _, _ = S.score(q, torch.stack([TA, TB, TB, TA]), [p["tgt_A"], p["tgt_B"], p["tgt_A"], p["tgt_B"]])
                ce["ext_EA_TA"], ce["ext_EB_TB"], ce["ext_EA_TB"], ce["ext_EB_TA"] = [-float(x) for x in sf]
                r = {"id": p["id"], "role": p["role"], "choice_valid": p["choice_valid"], "default": "A" if dA else "B", "donor": donor["id"],
                     "pref_A_minus_B": pref, "exec_ce_default": ce, "kd_self_default": float(kd(Zs, TA if dA else TB)), "kd_self_other": float(kd(Zs, TB if dA else TA)),
                     "zs_cos_TA_TB": [float(F.cosine_similarity(Zs.flatten(), TA.flatten(), 0)), float(F.cosine_similarity(Zs.flatten(), TB.flatten(), 0))]}
                if a.gen and (p["role"] == "heldout8" or p["choice_valid"]):
                    gens, _ = S.generate(q, Zall, a.gen_tokens)
                    r["gens"] = []
                    for nm, g_ in zip(names, gens):
                        pred = last_boxed(g_)
                        r["gens"].append({"cond": nm, "text": g_, "pred": pred, "correct": correct(pred, str(p["answer"])),
                                          "ended": "<|im_end|>" in g_, "exec_close": "</EXECUTION>" in g_, "n_chars": len(g_),
                                          "ov_A": ngram_overlap(g_, p["execution_A"]), "ov_B": ngram_overlap(g_, p["execution_B"]),
                                          "terms_A": method_hits(g_, p["method_terms_only_A"]), "terms_B": method_hits(g_, p["method_terms_only_B"])})
            prow.append(r)
            print(short(p["id"]), p["role"], "def", r["default"], {k: round(v, 3) for k, v in pref.items()}, flush=True)
        def psumm(R):
            s = {"n": len(R)}
            if not R:
                return s
            ext_ok = [r["pref_A_minus_B"]["teacher_A"] > 0 and r["pref_A_minus_B"]["teacher_B"] < 0 for r in R]
            s["external_4grid_both_correct"] = int(sum(ext_ok))
            s["external_TA_minus_TB_pref"] = {"mean": float(np.mean([r["pref_A_minus_B"]["teacher_A"] - r["pref_A_minus_B"]["teacher_B"] for r in R])),
                                              "ci95": boot([r["pref_A_minus_B"]["teacher_A"] - r["pref_A_minus_B"]["teacher_B"] for r in R]),
                                              "n_pos": int(sum(r["pref_A_minus_B"]["teacher_A"] > r["pref_A_minus_B"]["teacher_B"] for r in R))}
            sg = lambda r: 1 if r["default"] == "A" else -1
            oth = lambda r: "teacher_B" if r["default"] == "A" else "teacher_A"
            s["self_pref_toward_default_n"] = int(sum(sg(r) * r["pref_A_minus_B"]["self"] > 0 for r in R))
            v = [sg(r) * (r["pref_A_minus_B"]["self"] - r["pref_A_minus_B"][oth(r)]) for r in R]
            s["self_minus_otherplanZ_toward_default"] = {"mean": float(np.mean(v)), "ci95": boot(v), "n_pos": int(sum(x > 0 for x in v))}
            for c in ("crossq", "mean", "zero", "nolatent"):
                v = [sg(r) * (r["pref_A_minus_B"]["self"] - r["pref_A_minus_B"][c]) for r in R]
                s[f"self_minus_{c}_toward_default"] = {"mean": float(np.mean(v)), "ci95": boot(v), "n_pos": int(sum(x > 0 for x in v))}
                s[f"abs_pref_self_vs_{c}"] = float(np.mean([abs(r["pref_A_minus_B"]["self"] - r["pref_A_minus_B"][c]) for r in R]))
            for c in ("crossq", "mean", "zero"):
                v = [r["exec_ce_default"][c] - r["exec_ce_default"]["self"] for r in R]
                s[f"exec_ce_{c}_minus_self"] = {"mean": float(np.mean(v)), "ci95": boot(v)}
            v = [(r["exec_ce_default"]["ext_EA_TB"] - r["exec_ce_default"]["ext_EA_TA"] + r["exec_ce_default"]["ext_EB_TA"] - r["exec_ce_default"]["ext_EB_TB"]) / 2 for r in R]
            s["external_exec_ce_matched_gain"] = {"mean": float(np.mean(v)), "ci95": boot(v), "n_pos": int(sum(x > 0 for x in v))}
            s["kd_self_default_minus_other"] = float(np.mean([r["kd_self_default"] - r["kd_self_other"] for r in R]))
            G = [r for r in R if "gens" in r]
            if G:
                gg = lambda r, c: next(x for x in r["gens"] if x["cond"] == c)
                lab = lambda x: "A" if x["ov_A"] > x["ov_B"] else ("B" if x["ov_B"] > x["ov_A"] else "=")
                s["gen_n"] = len(G)
                for c in ("self", "crossq", "mean", "zero", "teacher_A", "teacher_B"):
                    s[f"gen_correct_{c}"] = int(sum(gg(r, c)["correct"] for r in G)); s[f"gen_ended_{c}"] = int(sum(gg(r, c)["ended"] and gg(r, c)["exec_close"] for r in G))
                    s[f"gen_auto_method_{c}"] = "".join(lab(gg(r, c)) for r in G)
                s["gen_self_auto_follows_default"] = int(sum(lab(gg(r, "self")) == r["default"] for r in G))
                s["gen_teacher_auto_switch"] = int(sum(lab(gg(r, "teacher_A")) == "A" and lab(gg(r, "teacher_B")) == "B" for r in G))
                s["gen_self_identical_crossq"] = int(sum(gg(r, "self")["text"] == gg(r, "crossq")["text"] for r in G))
                s["gen_self_identical_mean"] = int(sum(gg(r, "self")["text"] == gg(r, "mean")["text"] for r in G))
                s["gen_self_identical_otherplan"] = int(sum(gg(r, "self")["text"] == gg(r, "teacher_B" if r["default"] == "A" else "teacher_A")["text"] for r in G))
                s["gen_TA_identical_TB"] = int(sum(gg(r, "teacher_A")["text"] == gg(r, "teacher_B")["text"] for r in G))
            return s
        out["pairs"] = {}
        for role in ("heldout8", "train_pair"):
            for sub in ("all", "choice_valid"):
                R = [r for r in prow if r["role"] == role and (sub == "all" or r["choice_valid"])]
                out["pairs"][f"{role}/{sub}"] = psumm(R)
        os.makedirs(f"{a.out}/evaluations", exist_ok=True)
        json.dump(out, open(f"{a.out}/evaluations/summary_{name}.json", "w"), indent=1)
        with open(f"{a.out}/evaluations/rows_{name}.jsonl", "w") as f:
            for r in vrows:
                f.write(json.dumps({"kind": "val_single", **r}) + "\n")
            for r in prow:
                f.write(json.dumps({"kind": "pair", **r}) + "\n")
        if a.gen:
            with open(f"{a.out}/evaluations/generations_{name}.md", "w") as f:
                for r in prow:
                    if "gens" not in r:
                        continue
                    f.write(f"## {r['id']} ({r['role']}, default={r['default']}, choice_valid={r['choice_valid']})\n\n")
                    grp_ = {}
                    for x in r["gens"]:
                        grp_.setdefault(x["text"], []).append(x)
                    for text, xs in grp_.items():
                        x = xs[0]
                        f.write(f"- **{', '.join(y['cond'] for y in xs)}** correct={x['correct']} ov A/B={x['ov_A']:.2f}/{x['ov_B']:.2f} terms A/B={x['terms_A']}/{x['terms_B']}\n\n```\n{text[:900]}\n```\n\n")
        print(json.dumps({k: v for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
