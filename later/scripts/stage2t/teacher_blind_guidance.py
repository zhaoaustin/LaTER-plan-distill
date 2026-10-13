"""0-step Teacher-vs-Blind guidance audit (forward only, no_grad, no weights saved).

Same model / prompts / slot path as Stage 2C (anchorsim_2c.fwd + teacher_pre / student_pre) and the same 7 valid held-out questions, common
prefixes c and operation texts U_A / U_B as teacher_plan_audit.SPEC (00486 stays invalid).
Branches (all from Stage 2B joint step 500):  T[Z, A*] = teacher prompt (Q + answer hint) with latent Z in the 8 slots;
S_blind = student prompt (Q, no answer) with zeros in the 8 slots (= the Stage 2C blind branch).
Decision state = row k (k = token LCP of tok(c+U_A), tok(c+U_B)) -> next-token distribution after c + shared tokens; same text index in all branches.
  D_guide^s = JSD(T[Z_s, A*], S_blind) for s in {A, B, S};  D_answer = JSD(T[Z_zero, A*], S_blind);  D_AB^T;  D_SZ^T
  Delta_A / Delta_B (matched - swapped, per-token mean log p of U);  G_A / G_B (matched - teacher zero)
"""
import argparse
import csv
import json
import os
import statistics as st

import torch
import torch.nn.functional as F

from later.scripts.stage2t.anchorsim_2c import HINT, fwd, jsd, student_pre, teacher_pre
from later.scripts.stage2t.stage2c_long import sha256
from later.scripts.stage2t.teacher_plan_audit import INVALID_WHY, SPEC
from later.scripts.stage2t.three_step import TStudent, load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--s2b", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ckpt", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    ck = torch.load(a.ckpt, weights_only=False)
    assert ck["step"] == 500 and ck.get("phase") == "joint"
    M = TStudent(a.model, a.s2b, seed=0); M.load(ck["state"])
    for p in M.peft.parameters():
        p.requires_grad_(False)
    M.peft.eval()
    audit = {"ckpt": a.ckpt, "sha256": sha256(a.ckpt), "global_step": ck["step"], "phase": ck.get("phase"), "trainable_params": sum(p.numel() for p in M.peft.parameters() if p.requires_grad),
             "teacher_and_student_blind_share_weights": "yes: both are the Stage 2B step-500 snapshot (2C teacher = frozen copy of the init)",
             "hint_template": HINT, "z_teacher_source": f"{a.s2b}/data/z_cache.pt", "kv_cache": "none (use_cache=False, independent full forwards)", "grad": "torch.no_grad, eval()"}
    pairs = [p for p in load_jsonl(f"{a.s2b}/data/pairs.jsonl") if p["role"] == "heldout8"]
    rows = []
    for p in pairs:
        qid = p["id"][:11]; spec = SPEC[qid]
        base = {"id": p["id"], "method_A": p["method_label_A"], "method_B": p["method_label_B"]}
        if spec is None:
            rows.append(base | {"valid": False, "why_invalid": INVALID_WHY[qid]}); continue
        c, UA, UB = spec(p["execution_A"], p["execution_B"]); q, ans = p["question"], str(p["answer"])
        ia, ib = M.tok.encode(c + UA, add_special_tokens=False), M.tok.encode(c + UB, add_special_tokens=False)
        k = 0
        while k < min(len(ia), len(ib)) and ia[k] == ib[k]:
            k += 1
        tp, sp = teacher_pre(M, q, ans), student_pre(M, q)
        with torch.no_grad():
            Z = {"A": M.teacher(p["plan_A_norm"]), "B": M.teacher(p["plan_B_norm"]), "S": M.zlat(q).float().detach()}
            Z["zero"] = torch.zeros_like(Z["A"])
            dec, lpU, emb = {}, {}, {}
            for zn, z in Z.items():                                       # teacher branches
                for un, ids in (("A", ia), ("B", ib)):
                    lg, y, e = fwd(M, tp, z, ids, len(ids)); assert y.tolist() == ids
                    l = F.log_softmax(lg.float(), -1)
                    lpU[(zn, un)] = float(l[k:].gather(1, torch.tensor(ids[k:], device=M.dev).unsqueeze(1)).mean())
                    if un == "A":
                        dec["T_" + zn] = lg[k].float(); emb["T_" + zn] = e
            lb, yb, eb = fwd(M, sp, Z["zero"], ia, len(ia)); assert yb.tolist() == ia   # student blind (no answer)
            dec["S_blind"] = lb[k].float()
            lbB, _, _ = fwd(M, sp, Z["zero"], ib, len(ib))
            # alignment: blind decision row == same row in the B-text forward (identical prefix), teacher/student rows index the same text token k
            blind_row_consistent = float((lb[k] - lbB[k]).abs().max())
            nt, ns = len(tp), len(sp)
            slots_read = {zn: bool(torch.equal(emb["T_" + zn][0, nt:nt + 8], Z[zn].to(emb["T_" + zn].dtype))) for zn in Z}
            blind_slots_zero = bool((eb[0, ns:ns + 8] == 0).all())
            text_emb_same = bool(torch.equal(emb["T_A"][0, nt + 8:], eb[0, ns + 8:]))           # suffix + c + U embeddings identical across branches
        J = lambda x, y: float(jsd(dec[x].unsqueeze(0), dec[y].unsqueeze(0))[0])
        ent = lambda x: float(-(F.softmax(dec[x], -1) * F.log_softmax(dec[x], -1)).sum())
        r = base | {"valid": True, "k_common_tokens": k, "first_div_tok_A": M.tok.decode(ia[k:k + 1]), "first_div_tok_B": M.tok.decode(ib[k:k + 1]),
                    "c_tail": c[-80:], "U_A": UA, "U_B": UB,
                    "D_guide_A": J("T_A", "S_blind"), "D_guide_B": J("T_B", "S_blind"), "D_guide_S": J("T_S", "S_blind"), "D_answer": J("T_zero", "S_blind"),
                    "D_AB_T": J("T_A", "T_B"), "D_SZ_T": J("T_S", "T_zero"),
                    "Delta_A": lpU[("A", "A")] - lpU[("B", "A")], "Delta_B": lpU[("B", "B")] - lpU[("A", "B")],
                    "G_A": lpU[("A", "A")] - lpU[("zero", "A")], "G_B": lpU[("B", "B")] - lpU[("zero", "B")],
                    "s_UA": {zn: lpU[(zn, "A")] for zn in Z}, "s_UB": {zn: lpU[(zn, "B")] for zn in Z},
                    "p_first_A": {x: float(F.softmax(dec[x], -1)[ia[k]]) for x in dec}, "p_first_B": {x: float(F.softmax(dec[x], -1)[ib[k]]) for x in dec},
                    "top1": {x: M.tok.decode([int(dec[x].argmax())]) for x in dec}, "entropy": {x: ent(x) for x in dec},
                    "audit": {"slots_hold_Z": slots_read, "blind_slots_zero": blind_slots_zero, "text_embeddings_identical_T_vs_blind": text_emb_same,
                              "blind_row_A_vs_B_text_maxdiff": blind_row_consistent, "answer_in_teacher_prompt": ans in M.tok.decode(tp),
                              "answer_hint_in_student_prompt": "Reference final answer" in M.tok.decode(sp), "n_teacher_prompt": nt, "n_student_prompt": ns}}
        r["both_pos"] = r["Delta_A"] > 0 and r["Delta_B"] > 0
        rows.append(r)
        print(json.dumps({x: v for x, v in r.items() if x not in ("U_A", "U_B", "c_tail")}), flush=True)
    V = [r for r in rows if r["valid"]]
    summ = {}
    for x in ("D_guide_A", "D_guide_B", "D_guide_S", "D_answer", "D_AB_T", "D_SZ_T", "Delta_A", "Delta_B", "G_A", "G_B"):
        v = [r[x] for r in V]; summ[x] = {"mean": st.mean(v), "median": st.median(v), "n_pos": sum(t > 0 for t in v), "n": len(v)}
    summ["both_pos"] = sum(r["both_pos"] for r in V)
    summ["ratio_D_guide_S_over_D_answer"] = {"mean": st.mean(r["D_guide_S"] / r["D_answer"] for r in V), "per_q": [round(r["D_guide_S"] / r["D_answer"], 3) for r in V]}
    audit.update({"n_heldout": len(rows), "n_valid": len(V), "valid_ids": [r["id"] for r in V],
                  "all_slots_hold_Z": all(all(r["audit"]["slots_hold_Z"].values()) for r in V), "all_blind_slots_zero": all(r["audit"]["blind_slots_zero"] for r in V),
                  "all_text_embeddings_identical": all(r["audit"]["text_embeddings_identical_T_vs_blind"] for r in V),
                  "answer_only_in_teacher": all(r["audit"]["answer_in_teacher_prompt"] and not r["audit"]["answer_hint_in_student_prompt"] for r in V),
                  "max_blind_row_A_vs_B_text_diff": max(r["audit"]["blind_row_A_vs_B_text_maxdiff"] for r in V), "summary": summ})
    json.dump({"audit": audit, "rows": rows}, open(f"{a.out}/teacher_blind_guidance.json", "w"), indent=1)
    keys = ["id", "valid", "method_A", "method_B", "D_guide_A", "D_guide_B", "D_guide_S", "D_answer", "D_AB_T", "D_SZ_T", "Delta_A", "Delta_B", "both_pos", "G_A", "G_B",
            "first_div_tok_A", "first_div_tok_B", "k_common_tokens", "U_A", "U_B", "why_invalid"]
    with open(f"{a.out}/per_question.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore"); w.writeheader(); [w.writerow(r) for r in rows]
    print("AUDIT", json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
