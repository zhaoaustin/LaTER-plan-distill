"""0-step, forward-only audit: does the frozen answer-conditioned Stage 2C Teacher (= Stage 2B joint step 500) read the latent plan?

Teacher input = exactly the Stage 2C teacher branch: teacher_pre(Q + answer hint) + 8 latent slots + suffix + c + U   (anchorsim_2c.fwd, use_cache=False:
every condition is a fresh full forward, no KV reuse). Four latent conditions per question, everything else identical:
  Z_A^T, Z_B^T  = frozen Stage 2A step-300 encodings of the gold plans (2B z_cache, via TStudent.teacher)
  Z_S           = Stage 2B step-500 student free-running latents for Q (student prompt, no answer) = what 2C feeds the teacher
  Z_zero        = zeros in the 8 slots
c = one neutral common prefix per question (hand-specified below, identical in all 4 conditions); U_A / U_B = gold execution text of the first
method-specific operation.  Tokens: ids_X = tok(c + U_X); k = token LCP(ids_A, ids_B); U_X span = ids_X[k:]; decision row = position k.
s(U|Z) = mean log p over the U span.  Delta_A = s(U_A|Z_A) - s(U_A|Z_B); Delta_B = s(U_B|Z_B) - s(U_B|Z_A).
D_AB = JSD(p(.|Z_A), p(.|Z_B)) and D_SZ = JSD(p(.|Z_S), p(.|Z_zero)) at the decision row (full vocab, fp32).
"""
import argparse
import csv
import json
import math
import os

import torch
import torch.nn.functional as F

from later.scripts.stage2t.anchorsim_2c import HINT, fwd, jsd, teacher_pre
from later.scripts.stage2t.stage2c_long import sha256
from later.scripts.stage2t.three_step import TStudent, load_jsonl
from later.src.train.stage2_plan_dataset import render_prompt


def cut(text, start, end):
    """substring of text from the first occurrence of `start` through the end of the first following `end`."""
    i = text.index(start); j = text.index(end, i) + len(end)
    return text[i:j]


def upto(text, end):
    return text[:text.index(end) + len(end)]


# c / U specification (hand-audited; see the report for the reasoning per question).  None = invalid (no credible neutral common prefix).
SPEC = {
    "s1val-00336": lambda A, B: ("1. ", cut(A, "Use Lagrange", "x + y + z - 1 = 0 $."), cut(B, "Apply the Cauchy", "by choosing appropriate vectors.")),
    "s1val-00574": lambda A, B: ("1. ", cut(A, "Rewrite the equation", "(3^m)^2 $."), cut(B, "Test small values", "= 4 + 65 = 69 $")),
    "s1val-00569": lambda A, B: ("1. Recognize that the function is a rational function and aim to find its maximum value.\n2. ",
                                 cut(A, "Substitute $ y", "quadratic equation in $ x $:"), cut(B, "Compute the derivative using the quotient rule", "critical points.")),
    "s1val-00507": lambda A, B: (upto(A, "\\frac{\\pi}{3}$.  \n2. "), cut(A, "Recognize that the probability is the double integral", "over this region."),
                                 cut(B, "Recognize that the probability of", "evaluated at the lower bounds.")),
    "s1val-00479": lambda A, B: (upto(A, "(a^{2} - a) = 0\n$$  \n\n2. "), cut(A, "Recognize that the equation can be factored", "two expressions."),
                                 cut(B, "Identify the coefficients", "in standard form.")),
    "s1val-00309": lambda A, B: (upto(A, "4^{101} \\mod 9$.  \n3. "), cut(A, "Use Euler", "so $4^6 \\equiv 1 \\mod 9$."),
                                 cut(B, "Use modular exponentiation", "by reducing $101 \\mod 3$.")),
    "s1val-00305": lambda A, B: ("1. Use the first equation $ xy = 6 $ to ", cut(A, "express $ y $", "y = \\frac{6}{x}"), cut(B, "factor the second equation", "share it.")),
    "s1val-00486": None,
}
INVALID_WHY = {"s1val-00486": "both executions open with generic sentences ('use trigonometric identities' / 'angles are symmetric'); the first A-specific op "
                              "(arithmetic-progression identity) is mid step 1 and the first B-specific op (cos(2pi-x) rewrite -> 7th roots) is steps 2-4, "
                              "so no neutral common prefix ends at a decision point"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--s2b", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ckpt", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ck = torch.load(a.ckpt, weights_only=False)
    assert ck["step"] == 500 and ck.get("phase") == "joint"
    M = TStudent(a.model, a.s2b, seed=0); M.load(ck["state"])
    for p in M.peft.parameters():
        p.requires_grad_(False)
    M.peft.eval()
    audit = {"teacher_ckpt": a.ckpt, "teacher_sha256": sha256(a.ckpt), "ckpt_global_step": ck["step"], "phase": ck.get("phase"),
             "trainable_params": sum(p.numel() for p in M.peft.parameters() if p.requires_grad), "hint_template": HINT,
             "z_teacher_source": f"{a.s2b}/data/z_cache.pt", "kv_cache": "none: every condition is an independent full forward (use_cache=False)"}
    pairs = [p for p in load_jsonl(f"{a.s2b}/data/pairs.jsonl") if p["role"] == "heldout8"]
    rows, details = [], []
    for p in pairs:
        qid = p["id"][:11]; spec = SPEC[qid]
        base = {"id": p["id"], "method_A": p["method_label_A"], "method_B": p["method_label_B"]}
        if spec is None:
            rows.append(base | {"valid": False, "why_invalid": INVALID_WHY[qid]}); continue
        A, B = p["execution_A"], p["execution_B"]
        c, UA, UB = spec(A, B)
        q, ans = p["question"], str(p["answer"])
        ia, ib = M.tok.encode(c + UA, add_special_tokens=False), M.tok.encode(c + UB, add_special_tokens=False)
        k = 0
        while k < min(len(ia), len(ib)) and ia[k] == ib[k]:
            k += 1
        sA, sB = ia[k:], ib[k:]
        pre = teacher_pre(M, q, ans)
        with torch.no_grad():
            Z = {"A": M.teacher(p["plan_A_norm"]), "B": M.teacher(p["plan_B_norm"]), "S": M.zlat(q).float()}
            Z["zero"] = torch.zeros_like(Z["A"])
            lp, dec, embs = {}, {}, {}
            for zn, z in Z.items():
                for un, ids, span in (("A", ia, sA), ("B", ib, sB)):
                    lg, y, emb = fwd(M, pre, z, ids, len(ids))          # rows predict ids[t]
                    assert y.tolist() == ids
                    l = F.log_softmax(lg.float(), -1)
                    lp[(zn, un)] = l[k:, :].gather(1, torch.tensor(span, device=M.dev).unsqueeze(1)).squeeze(1)
                    if un == "A":
                        dec[zn] = lg[k].float(); embs[zn] = emb
            # input / slot audit
            n = len(pre); sl = slice(n, n + 8)
            slot_ok = all(torch.equal(embs[zn][0, sl], Z[zn].to(embs[zn].dtype)) for zn in Z)
            nonslot_same = all(torch.equal(embs[zn][0, :n], embs["A"][0, :n]) and torch.equal(embs[zn][0, n + 8:], embs["A"][0, n + 8:]) for zn in Z)
        s = lambda zn, un: float(lp[(zn, un)].mean())
        dA, dB = s("A", "A") - s("B", "A"), s("B", "B") - s("A", "B")
        pa = lambda zn: math.exp(float(lp[(zn, "A")][0])); pb = lambda zn: math.exp(float(lp[(zn, "B")][0]))
        J = lambda x, y: float(jsd(dec[x].unsqueeze(0), dec[y].unsqueeze(0))[0])
        cosf = lambda x, y: float(F.cosine_similarity(Z[x].flatten(), Z[y].flatten(), 0))
        cos_slot = lambda x, y: [round(float(v), 4) for v in F.cosine_similarity(Z[x], Z[y], -1)]
        prompt_txt = M.tok.decode(pre)
        leak = {"answer_in_prompt": ans in prompt_txt, "plan_A_text_in_input": p["plan_A_norm"][:60] in prompt_txt + c,
                "plan_B_text_in_input": p["plan_B_norm"][:60] in prompt_txt + c,
                "student_prompt_has_answer": "Reference final answer" in M.tok.decode(M.prompt_ids(q))}
        r = base | {"valid": True, "c": c, "U_A": UA, "U_B": UB, "n_common_tokens_k": k, "len_UA_tok": len(sA), "len_UB_tok": len(sB),
                    "first_div_tok_A": M.tok.decode(sA[:1]), "first_div_tok_B": M.tok.decode(sB[:1]),
                    "s_UA_ZA": s("A", "A"), "s_UA_ZB": s("B", "A"), "s_UB_ZB": s("B", "B"), "s_UB_ZA": s("A", "B"),
                    "s_UA_ZS": s("S", "A"), "s_UA_Z0": s("zero", "A"), "s_UB_ZS": s("S", "B"), "s_UB_Z0": s("zero", "B"),
                    "Delta_A": dA, "Delta_B": dB, "both_pos": dA > 0 and dB > 0,
                    "p_first_tokA_ZA": pa("A"), "p_first_tokA_ZB": pa("B"), "p_first_tokB_ZB": pb("B"), "p_first_tokB_ZA": pb("A"),
                    "first_tok_matched_minus_swapped_A": pa("A") - pa("B"), "first_tok_matched_minus_swapped_B": pb("B") - pb("A"),
                    "p_first_tokA_ZS": pa("S"), "p_first_tokA_Z0": pa("zero"), "p_first_tokB_ZS": pb("S"), "p_first_tokB_Z0": pb("zero"),
                    "D_AB": J("A", "B"), "D_SZ": J("S", "zero"), "D_A0": J("A", "zero"), "D_B0": J("B", "zero"),
                    "norm_ZA": float(Z["A"].norm(dim=-1).mean()), "norm_ZB": float(Z["B"].norm(dim=-1).mean()), "norm_ZS": float(Z["S"].norm(dim=-1).mean()),
                    "cos_ZA_ZB": cosf("A", "B"), "cos_ZA_ZS": cosf("A", "S"), "cos_ZB_ZS": cosf("B", "S"),
                    "cos_slot_ZA_ZB": cos_slot("A", "B"), "cos_slot_ZA_ZS": cos_slot("A", "S"), "cos_slot_ZB_ZS": cos_slot("B", "S"),
                    "norm_slot_ZA": [round(float(v), 4) for v in Z["A"].norm(dim=-1)], "norm_slot_ZS": [round(float(v), 4) for v in Z["S"].norm(dim=-1)],
                    "shapes": {zn: list(z.shape) for zn, z in Z.items()}, "slots_hold_Z": slot_ok, "nonslot_identical_4conds": nonslot_same, **leak}
        rows.append(r)
        print(json.dumps({k_: v for k_, v in r.items() if not k_.startswith(("cos_slot", "norm_slot", "c", "U_"))}), flush=True)
    V = [r for r in rows if r["valid"]]
    mean = lambda k_: sum(r[k_] for r in V) / len(V)
    audit.update({"n_heldout": len(rows), "n_valid": len(V), "all_slots_hold_Z": all(r["slots_hold_Z"] for r in V), "all_nonslot_identical": all(r["nonslot_identical_4conds"] for r in V),
                  "answer_in_teacher_prompt_all": all(r["answer_in_prompt"] for r in V), "no_plan_text_in_input": not any(r["plan_A_text_in_input"] or r["plan_B_text_in_input"] for r in V),
                  "student_prompt_no_answer": not any(r["student_prompt_has_answer"] for r in V),
                  "summary": {"both_pos": sum(r["both_pos"] for r in V), "Delta_A_pos": sum(r["Delta_A"] > 0 for r in V), "Delta_B_pos": sum(r["Delta_B"] > 0 for r in V),
                              "mean_Delta_A": mean("Delta_A"), "mean_Delta_B": mean("Delta_B"), "mean_D_AB": mean("D_AB"), "mean_D_SZ": mean("D_SZ"),
                              "mean_D_A0": mean("D_A0"), "mean_D_B0": mean("D_B0"), "mean_norm_ZA": mean("norm_ZA"), "mean_norm_ZS": mean("norm_ZS"),
                              "mean_cos_ZA_ZB": mean("cos_ZA_ZB"), "mean_cos_ZA_ZS": mean("cos_ZA_ZS"), "mean_cos_ZB_ZS": mean("cos_ZB_ZS")}})
    json.dump({"audit": audit, "rows": rows}, open(f"{a.out}/teacher_plan_audit.json", "w"), indent=1)
    keys = ["id", "valid", "method_A", "method_B", "first_div_tok_A", "first_div_tok_B", "n_common_tokens_k", "len_UA_tok", "len_UB_tok", "Delta_A", "Delta_B", "both_pos",
            "first_tok_matched_minus_swapped_A", "first_tok_matched_minus_swapped_B", "D_AB", "D_SZ", "D_A0", "D_B0", "s_UA_ZS", "s_UA_Z0", "s_UB_ZS", "s_UB_Z0",
            "norm_ZA", "norm_ZB", "norm_ZS", "cos_ZA_ZB", "cos_ZA_ZS", "cos_ZB_ZS", "U_A", "U_B", "why_invalid"]
    with open(f"{a.out}/per_question.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore"); w.writeheader(); [w.writerow(r) for r in rows]
    print("AUDIT", json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
