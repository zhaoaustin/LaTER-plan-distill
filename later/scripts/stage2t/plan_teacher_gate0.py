"""Gate 0 (forward only): does the frozen Stage 2B step-500 Teacher prefer the matching first math operation when its privileged
condition is the verified explicit Plan text (P_A vs P_B) instead of the answer?

Teacher prompt = plan_teacher_pre(Q, P) : the Plan placed exactly where the 2C answer hint was (end of the user turn), same wrapper style.
Latent slots hold the SAME fixed Z_ref in both conditions (Z_ref = Stage 2B step-500 student Z_S(Q); robustness: zeros). Same 7 valid
questions, c and U_A / U_B as teacher_plan_audit.SPEC.  Delta_A = s(U_A|P_A) - s(U_A|P_B), Delta_B = s(U_B|P_B) - s(U_B|P_A).
Pre-registered gate: >= 5/7 questions with Delta_A > 0.01 and Delta_B > 0.01 (nats/token) under Z_ref = Z_S.
"""
import argparse
import csv
import json
import os
import re

import torch
import torch.nn.functional as F

from later.scripts.stage2t.anchorsim_2c import fwd, jsd
from later.scripts.stage2t.stage2c_long import sha256
from later.scripts.stage2t.teacher_plan_audit import INVALID_WHY, SPEC
from later.scripts.stage2t.three_step import TStudent, load_jsonl
from later.src.train.stage2_plan_dataset import render_prompt

PLAN_HINT = "\n\n(Reference plan, visible to the teacher only:\n{})"
GATE_MIN_BOTH, GATE_EPS = 5, 0.01


def plan_teacher_pre(M, q, plan):
    return M.tok.encode(render_prompt(M.tok, q + PLAN_HINT.format(plan)), add_special_tokens=False) + M.T.plan_open_ids


def answer_in_plan(plan, ans):
    a = str(ans).strip()
    if len(a) <= 2:                                            # short answers: only count explicit "= a" / "is a" / boxed forms
        return bool(re.search(r"(=|is|\\boxed\{)\s*\$?\s*" + re.escape(a) + r"(?![\w.])", plan))
    return a in plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--s2b", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ckpt", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(0)
    ck = torch.load(a.ckpt, weights_only=False); assert ck["step"] == 500 and ck.get("phase") == "joint"
    M = TStudent(a.model, a.s2b, seed=0); M.load(ck["state"])
    for p in M.peft.parameters():
        p.requires_grad_(False)
    M.peft.eval()
    audit = {"ckpt": a.ckpt, "sha256": sha256(a.ckpt), "global_step": ck["step"], "trainable_params": sum(p.numel() for p in M.peft.parameters() if p.requires_grad),
             "plan_hint_template": PLAN_HINT, "gate": f">= {GATE_MIN_BOTH}/7 with Delta_A > {GATE_EPS} and Delta_B > {GATE_EPS} (Z_ref = Z_S)"}
    rows = []
    for p in [p for p in load_jsonl(f"{a.s2b}/data/pairs.jsonl") if p["role"] == "heldout8"]:
        qid = p["id"][:11]; spec = SPEC[qid]; base = {"id": p["id"], "method_A": p["method_label_A"], "method_B": p["method_label_B"]}
        if spec is None:
            rows.append(base | {"valid": False, "why_invalid": INVALID_WHY[qid]}); continue
        c, UA, UB = spec(p["execution_A"], p["execution_B"]); q = p["question"]
        ia, ib = M.tok.encode(c + UA, add_special_tokens=False), M.tok.encode(c + UB, add_special_tokens=False)
        k = 0
        while k < min(len(ia), len(ib)) and ia[k] == ib[k]:
            k += 1
        pre = {"PA": plan_teacher_pre(M, q, p["plan_A_norm"]), "PB": plan_teacher_pre(M, q, p["plan_B_norm"])}
        r = base | {"valid": True, "k": k, "first_div_tok_A": M.tok.decode(ia[k:k + 1]), "first_div_tok_B": M.tok.decode(ib[k:k + 1]), "U_A": UA, "U_B": UB,
                    "answer_in_plan_A": answer_in_plan(p["plan_A_norm"], p["answer"]), "answer_in_plan_B": answer_in_plan(p["plan_B_norm"], p["answer"]),
                    "answer_hint_absent": all("Reference final answer" not in M.tok.decode(v) for v in pre.values())}
        with torch.no_grad():
            Zs = {"ZS": M.zlat(q).float()}; Zs["Z0"] = torch.zeros_like(Zs["ZS"])
            for zn, z in Zs.items():
                s, dec = {}, {}
                for pn, pp in pre.items():
                    for un, ids in (("A", ia), ("B", ib)):
                        lg, y, e = fwd(M, pp, z, ids, len(ids)); assert y.tolist() == ids
                        n = len(pp); assert torch.equal(e[0, n:n + 8], z.to(e.dtype))
                        l = F.log_softmax(lg.float(), -1)
                        s[(pn, un)] = float(l[k:].gather(1, torch.tensor(ids[k:], device=M.dev).unsqueeze(1)).mean())
                        if un == "A":
                            dec[pn] = lg[k].float()
                dA, dB = s[("PA", "A")] - s[("PB", "A")], s[("PB", "B")] - s[("PA", "B")]
                pf = lambda pn, ids: float(F.softmax(dec[pn], -1)[ids[k]])
                r[zn] = {"Delta_A": dA, "Delta_B": dB, "both_pos": dA > 0 and dB > 0, "both_gt_eps": dA > GATE_EPS and dB > GATE_EPS,
                         "s": {f"{u}|{pn}": v for (pn, u), v in s.items()}, "JSD_PA_PB_decision": float(jsd(dec["PA"].unsqueeze(0), dec["PB"].unsqueeze(0))[0]),
                         "p_firstA": {pn: pf(pn, ia) for pn in dec}, "p_firstB": {pn: pf(pn, ib) for pn in dec},
                         "top1": {pn: M.tok.decode([int(dec[pn].argmax())]) for pn in dec}}
        rows.append(r); print(json.dumps({x: v for x, v in r.items() if x not in ("U_A", "U_B")}), flush=True)
    V = [r for r in rows if r["valid"]]
    for zn in ("ZS", "Z0"):
        audit[f"summary_{zn}"] = {"both_pos": sum(r[zn]["both_pos"] for r in V), "both_gt_eps": sum(r[zn]["both_gt_eps"] for r in V),
                                  "Delta_A_pos": sum(r[zn]["Delta_A"] > 0 for r in V), "Delta_B_pos": sum(r[zn]["Delta_B"] > 0 for r in V),
                                  "mean_Delta_A": sum(r[zn]["Delta_A"] for r in V) / len(V), "mean_Delta_B": sum(r[zn]["Delta_B"] for r in V) / len(V),
                                  "mean_JSD": sum(r[zn]["JSD_PA_PB_decision"] for r in V) / len(V)}
    audit["n_valid"] = len(V); audit["GATE0_PASS"] = audit["summary_ZS"]["both_gt_eps"] >= GATE_MIN_BOTH
    audit["answer_leak_in_plans"] = {r["id"]: [r["answer_in_plan_A"], r["answer_in_plan_B"]] for r in V}
    json.dump({"audit": audit, "rows": rows}, open(f"{a.out}/gate0.json", "w"), indent=1)
    with open(f"{a.out}/gate0_per_question.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "valid", "method_A", "method_B", "first_div_A", "first_div_B", "ZS_Delta_A", "ZS_Delta_B", "ZS_both_gt_eps", "Z0_Delta_A", "Z0_Delta_B",
                                       "JSD_PA_PB_ZS", "answer_in_PA", "answer_in_PB", "U_A", "U_B"])
        for r in rows:
            w.writerow([r["id"], r["valid"], r["method_A"], r["method_B"]] + ([r["first_div_tok_A"], r["first_div_tok_B"], r["ZS"]["Delta_A"], r["ZS"]["Delta_B"], r["ZS"]["both_gt_eps"],
                       r["Z0"]["Delta_A"], r["Z0"]["Delta_B"], r["ZS"]["JSD_PA_PB_decision"], r["answer_in_plan_A"], r["answer_in_plan_B"], r["U_A"], r["U_B"]] if r["valid"] else [""] * 12))
    print("AUDIT", json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
