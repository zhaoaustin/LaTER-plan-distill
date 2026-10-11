"""P1 (forward-only): PlanRec CE restricted to frozen plan-span categories, learned vs shuffled (3 derangements) vs
mean vs zero latents. Same records / plan / teacher-forced prefix / targets / masks as the earlier Test 2; only Z
changes. Token tags come from plan_span_rules (frozen before evaluation). model.eval() + torch.inference_mode()."""
import argparse
import json
import os
import random
import statistics as st

import pandas as pd
import torch

from later.scripts.data.audit_plan_specificity import token_tags
from later.scripts.data.plan_span_rules import build_generic_ngrams, label_chars
from later.src.train.stage2_common import dump_json, load_stage2_model, load_tokenizer
from later.src.train.stage2_plan_dataset import Stage2Collator, build_views, stage2_tokens
from later.src.train.stage2_planrec_monitor import build_latents, fixed_derangement

COND = ["learned", "shuffled_s0", "shuffled_s1", "shuffled_s2", "mean", "zero"]
CATS = {"full": None, "generic": {"generic_template"}, "method": {"method_specific"}, "subgoal": {"subgoal_specific"},
        "specific_combined": {"method_specific", "subgoal_specific"}, "unknown": {"unknown"}}


def boot(d, w, n=2000, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(d)))
    stat = lambda ii: sum(d[i] * w[i] for i in ii) / max(sum(w[i] for i in ii), 1e-12)
    bs = sorted(stat([rng.choice(idx) for _ in idx]) for _ in range(n))
    return {"estimate": stat(idx), "ci95": [bs[int(0.025 * n)], bs[int(0.975 * n) - 1]], "n_records": len(d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = load_tokenizer(a.stage1)
    T = stage2_tokens(tok)
    gen = build_generic_ngrams(list(pd.read_parquet(f"{a.data_dir}/stage2_train.parquet").plan))
    frozen = json.load(open(os.path.join(a.out, "span_rules_frozen.json")))
    assert frozen["generic_ngrams"] == gen, "span rules differ from the frozen P0 rules"
    val = pd.read_parquet(f"{a.data_dir}/stage2_val.parquet")
    rows = [val.iloc[i].to_dict() for i in range(a.n)]
    tags = {}
    validation = {"records": a.n, "target_ids_match_offset_tokenization": True, "decode_checks": []}
    for r in rows:
        lab = label_chars(r["plan"], gen)
        tgt, offs, tt = token_tags(tok, T, r, lab)
        tags[r["record_id"]] = (tgt, tt)
    model = load_stage2_model(a.model, latent_embed_builder_no_grad=True, latent_projector_dropout=0.0,
                              dtype=torch.float32, latent_output_rms="token_embedding").cuda().eval()
    col = Stage2Collator(tok.pad_token_id)
    batches = [col([build_views(tok, r, 8, T, 4096) for r in rows[s: s + 8]]) for s in range(0, a.n, 8)]
    dev = "cuda"
    per_tok = {}   # rid -> cond -> list of nll
    with torch.inference_mode():
        Z = torch.cat([build_latents(model, b) for b in batches])
        perms = {f"shuffled_s{s}": fixed_derangement(a.n, seed=s) for s in range(3)}
        zmean = Z.mean(0)
        off = 0
        for b in batches:
            B = b["student_input_ids"].size(0)
            idx = list(range(off, off + B))
            off += B
            for cond in COND:
                z = (Z[idx] if cond == "learned" else torch.zeros_like(Z[idx]) if cond == "zero" else
                     zmean.unsqueeze(0).expand(B, -1, -1) if cond == "mean" else Z[[perms[cond][i] for i in idx]])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    rec = model.forward_plan_reconstruction(z, b["planrec_input_ids"].to(dev), b["planrec_attention_mask"].to(dev),
                                                            b["planrec_labels"].to(dev))
                cnt = rec["target_mask"].sum(1).tolist()
                tgt_flat = b["planrec_labels"][:, 1:].to(dev)[rec["target_mask"]].tolist()
                s = 0
                for i in range(B):
                    rid = b["record_ids"][i]
                    if tgt_flat[s: s + cnt[i]] != tags[rid][0]:
                        validation["target_ids_match_offset_tokenization"] = False
                    per_tok.setdefault(rid, {})[cond] = rec["token_nll"][s: s + cnt[i]].double().cpu().tolist()
                    s += cnt[i]
    # decode checks: tokens tagged specific decode to text inside the plan
    for r in rows[:12]:
        tgt, tt = tags[r["record_id"]]
        spec = [tok.decode([t]) for t, g in zip(tgt, tt) if g in ("method_specific", "subgoal_specific")]
        validation["decode_checks"].append({"record_id": r["record_id"], "n_target_tokens": len(tgt),
                                            "n_specific_tokens": len(spec), "all_specific_tokens_in_plan": all(x.strip() in r["plan"] for x in spec),
                                            "specific_tokens_head": spec[:25]})
    validation["token_counts_equal_across_conditions"] = all(len({len(v[c]) for c in COND}) == 1 for v in per_tok.values())
    validation["shift_rule"] = "token_nll[j] = -log p(target_j | Z, '<EXPLORATION>\\n', plan targets < j); target_0 predicted from the last prefix position"
    validation["prev_test2_full_ce_learned"] = 0.70696
    # aggregate
    ids = sorted(per_tok)
    res = {"model": a.model, "n_records": len(ids), "conditions": {}, "gains": {}, "coverage": {}}
    def sums(rid, cond, cat):
        tt = tags[rid][1]
        v = [x for x, g in zip(per_tok[rid][cond], tt) if CATS[cat] is None or g in CATS[cat]]
        return sum(v), len(v)
    tot = sum(len(tags[i][1]) for i in ids)
    for cat in CATS:
        n_tok = sum(sums(i, "learned", cat)[1] for i in ids)
        n_rec = sum(1 for i in ids if sums(i, "learned", cat)[1] > 0)
        res["coverage"][cat] = {"tokens": n_tok, "token_share": n_tok / tot, "records_with_tokens": n_rec}
        for cond in COND:
            S = [sums(i, cond, cat) for i in ids]
            res["conditions"].setdefault(cond, {})[f"ce_{cat}"] = sum(x for x, _ in S) / max(1, sum(n for _, n in S))
        valid = [i for i in ids if sums(i, "learned", cat)[1] > 0]
        for cond in COND[1:]:
            dtw = [(sums(i, cond, cat)[0] - sums(i, "learned", cat)[0]) / sums(i, "learned", cat)[1] for i in valid]
            w = [sums(i, "learned", cat)[1] for i in valid]
            g_tw = boot(dtw, w)
            g_pq = boot(dtw, [1.0] * len(valid))
            g_pq.update({"mean": st.mean(dtw), "median": st.median(dtw), "frac_positive": sum(x > 0 for x in dtw) / len(dtw)})
            res["gains"][f"{cat}_gain_{cond}"] = {"token_weighted": g_tw, "per_question": g_pq}
        # 3-seed shuffled average per record
        dtw = [sum(sums(i, f"shuffled_s{s}", cat)[0] - sums(i, "learned", cat)[0] for s in range(3)) / 3 / sums(i, "learned", cat)[1] for i in valid]
        w = [sums(i, "learned", cat)[1] for i in valid]
        g = boot(dtw, [1.0] * len(valid))
        g.update({"mean": st.mean(dtw), "median": st.median(dtw), "frac_positive": sum(x > 0 for x in dtw) / len(dtw)})
        res["gains"][f"{cat}_gain_shuffled_3seed"] = {"token_weighted": boot(dtw, w), "per_question": g}
    validation["full_ce_learned_reproduces_test2"] = abs(res["conditions"]["learned"]["ce_full"] - 0.70696) < 2e-4
    dump_json(res, os.path.join(a.out, "p1_conditionality_metrics.json"))
    dump_json(validation, os.path.join(a.out, "p1_eval_validation.json"))
    out = []
    for i in ids:
        for cond in COND:
            d = {"record_id": i, "condition": cond}
            for cat in CATS:
                s, n = sums(i, cond, cat)
                d[f"ce_{cat}"] = s / n if n else None
                d[f"tokens_{cat}"] = n
            out.append(d)
    pd.DataFrame(out).to_csv(os.path.join(a.out, "p1_per_example.csv"), index=False)
    # sanity: specific-token log-probs, learned vs shuffled_s0, first 15 records with >= 10 specific tokens
    with open(os.path.join(a.out, "p1_specific_token_examples.md"), "w") as f:
        f.write("# P1 sanity: specific-token log p, learned vs shuffled_s0 (B step 300). Format: token: learned / shuffled\n\n")
        k = 0
        for r in rows:
            rid = r["record_id"]
            tgt, tt = tags[rid]
            sel = [j for j, g in enumerate(tt) if g in ("method_specific", "subgoal_specific")]
            if len(sel) < 10:
                continue
            L, S = per_tok[rid]["learned"], per_tok[rid]["shuffled_s0"]
            f.write(f"## {rid}\n\nplan (first 400 chars): `{r['plan'][:400]!r}`\n\n")
            f.write(" | ".join(f"`{tok.decode([tgt[j]])}`[{tt[j][0]}]: {-L[j]:.3f} / {-S[j]:.3f}" for j in sel[:30]) + "\n\n")
            f.write(f"mean specific log p: learned {-sum(L[j] for j in sel) / len(sel):.4f}, shuffled {-sum(S[j] for j in sel) / len(sel):.4f}\n\n")
            k += 1
            if k == 15:
                break
    print("P1_DONE", json.dumps({c: {k2: round(v, 5) for k2, v in res["conditions"][c].items()} for c in ("learned", "shuffled_s0", "mean")}),
          json.dumps({k2: v for k2, v in validation.items() if k2 != "decode_checks"}))


if __name__ == "__main__":
    main()
