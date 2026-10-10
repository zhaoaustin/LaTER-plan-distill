"""Forward-only diagnostics for OUR Stage 2 (no parameter updates, no loss changes).

Fixed set: the first N records of stage2_val (same records as the training probe). Execution targets are the
gold E (+closing) tokens; "nonrep" drops tokens on lines that restate a plan line verbatim.

Test A (Stage-2 ckpt)  learned / zero / shuffled latents (same L, positions, boundary): execution CE, next-token
                       KL(learned || zero), KL(learned || shuffled), top-1 agreement, by execution position bucket;
                       per-layer boundary alignment (normalized L1 to the frozen teacher) for each condition.
Test B (Stage-1 teacher) plan A (gold) vs B (same steps, reversed order), M (another record's plan), N (empty):
                       execution CE of E_A, next-token KL(A || variant), per-layer teacher boundary distance
                       d_l(A, variant) (same normalized L1 as L_alignment); greedy generation with A vs B to measure
                       strategy switching (does the execution follow the given step order?).
Test C (Stage-2 + Stage-1) boundary-to-execution transfer: overwrite the boundary-position residual stream with the
                       teacher's plan-informed boundary hidden (activation patching, chosen layers) and measure how much
                       of the gap to the gold-plan execution CE closes.
"""
import argparse
import json
import os
import re
import time
from contextlib import contextmanager

import pandas as pd
import torch
import torch.nn.functional as F

from later.src.train.stage2_common import dump_json, load_stage2_model, load_tokenizer
from later.src.train.stage2_plan_dataset import (EOS, EXEC_CLOSE, PLAN_OPEN, SEPARATOR, Stage2Collator, build_views,
                                                 render_prompt, stage2_tokens)
from later.src.train.stage2_teacher_cache import TeacherBoundaryCache

LAYERS = list(range(1, 37))
BUCKETS = [(0, 16), (16, 64), (64, 10 ** 9)]


# ------------------------------------------------------------------ shared forward utilities
@contextmanager
def patch_boundary(model, positions, vectors):
    """Replace decoder-layer-l output at positions[b] by vectors[l][b] (only for layers in vectors)."""
    handles = []
    for lid, vec in vectors.items():
        def hook(_m, _a, out, vec=vec):
            h = out[0] if isinstance(out, (tuple, list)) else out
            h = h.clone()
            rows = torch.arange(h.size(0), device=h.device)
            h[rows, positions.to(h.device)] = vec.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, (tuple, list)) else h
        handles.append(model.model.layers[lid - 1].register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def latent_embeds(model, b, mode="learned"):
    """Run LaTER's builder (no grad) and return full input embeddings for the student latent path."""
    dev = "cuda"
    ids = b["student_input_ids"].to(dev)
    tok_emb = model.get_input_embeddings()(ids)
    over, emask, _ = model._build_latent_recursive_embeds(
        input_ids=ids, attention_mask=b["student_attention_mask"].to(dev), position_ids=b["student_position_ids"].to(dev),
        cot_mask=None, latent_positions=b["latent_positions"].to(dev), latent_slot_mask=None,
        latent_start_positions=b["latent_start_positions"].to(dev),
        global_prompt_len=int(b["latent_start_positions"].max()) + 1, use_no_grad=True)
    if mode == "zero":
        over = torch.zeros_like(over)
    elif mode == "shuffled":
        rows = torch.arange(ids.size(0), device=dev).unsqueeze(1)
        lp = b["latent_positions"].to(dev)
        over = over.clone()
        over[rows, lp] = torch.roll(over[rows, lp], -1, 0)
    return torch.where(emask.unsqueeze(-1), over.to(tok_emb.dtype), tok_emb)


def run(model, input_ids=None, embeds=None, mask=None, pos=None, capture_pos=None, patch=None):
    """Full forward; returns last hidden and boundary hidden per layer (captured after patching)."""
    store = {}
    ctx_patch = patch_boundary(model, capture_pos, patch) if patch else _null()
    with ctx_patch, model.capture_layer_outputs_at(capture_pos.cuda(), LAYERS) as (st, _):
        out = model.model(input_ids=None if embeds is not None else input_ids.cuda(), inputs_embeds=embeds,
                          attention_mask=mask.cuda(), position_ids=pos.cuda(), use_cache=False, return_dict=True)
    store.update(st)
    return out.last_hidden_state, {l: st[l] for l in LAYERS}


@contextmanager
def _null():
    yield


def exec_logprobs(model, hidden, labels, row):
    """log-softmax over vocab at every position predicting a supervised label in `row` -> [n_tok, V] (fp32)."""
    tgt = labels[row, 1:].cuda()
    sel = tgt != -100
    h = hidden[row, :-1][sel]
    return F.log_softmax(model.lm_head(h).float(), dim=-1), tgt[sel]


def norm_l1(hs, ht):
    hs, ht = hs.float(), ht.float()
    return (hs - ht).abs().sum(-1) / (hs.shape[-1] * (ht.std(dim=-1, correction=0) + 1e-6))


class Acc:
    def __init__(self):
        self.d = {}

    def add(self, key, value, n=1):
        s, c = self.d.get(key, (0.0, 0))
        self.d[key] = (s + float(value), c + n)

    def out(self):
        return {k: s / c for k, (s, c) in self.d.items() if c}


def rep_mask(seg_r, labels, row):
    tgt = labels[row, 1:]
    return seg_r[row, 1:][tgt != -100].cuda()


# ------------------------------------------------------------------ Test A + C (Stage-2 checkpoint)
@torch.no_grad()
def test_ac(args, tok, frame, cache):
    T = stage2_tokens(tok)
    collate = Stage2Collator(tok.pad_token_id)
    m2 = load_stage2_model(args.stage2, latent_embed_builder_no_grad=True, latent_projector_dropout=0.0,
                           dtype=torch.bfloat16).cuda().eval()
    A, C, align = Acc(), Acc(), {c: Acc() for c in ("learned", "zero", "shuffled", "explicit_student")}
    patch_sets = {"all": LAYERS, "early_1-12": LAYERS[:12], "mid_13-24": LAYERS[12:24], "late_25-36": LAYERS[24:],
                  **{f"layer_{l}": [l] for l in (1, 9, 18, 27, 36)}}
    rows_all = [frame.iloc[i].to_dict() for i in range(len(frame))]
    for s in range(0, len(rows_all), args.batch_size):
        chunk = rows_all[s: s + args.batch_size]
        if len(chunk) < 2:
            break
        b = collate([build_views(tok, r, args.latent_steps, T, 4096) for r in chunk])
        th = cache.get(b["record_ids"], LAYERS, "cuda")
        mask, pos, bpos, lab = b["student_attention_mask"], b["student_position_ids"], b["student_boundary"], b["student_labels"]
        hid, bnd, lp = {}, {}, {}
        for cond in ("learned", "zero", "shuffled"):
            hid[cond], bnd[cond] = run(m2, embeds=latent_embeds(m2, b, cond), mask=mask, pos=pos, capture_pos=bpos)
        # Stage-2 student on the explicit (gold-plan) path: reference distribution on the same E tokens
        hx, bx = run(m2, input_ids=b["explicit_input_ids"], mask=b["explicit_attention_mask"],
                     pos=b["explicit_position_ids"], capture_pos=b["explicit_boundary"])
        xlab = b["explicit_labels"].clone()
        keep = b["explicit_seg_E"] | b["explicit_seg_A"] | b["explicit_seg_C"]
        xlab[~keep] = -100
        # Test C patched runs (learned latents + teacher boundary vector at chosen layers)
        patched = {name: run(m2, embeds=latent_embeds(m2, b, "learned"), mask=mask, pos=pos, capture_pos=bpos,
                             patch={l: th[l] for l in ls})[0] for name, ls in patch_sets.items()}
        for r in range(len(chunk)):
            nonrep = ~rep_mask(b["student_seg_R"], lab, r)
            for cond in ("learned", "zero", "shuffled"):
                lp[cond], tgt = exec_logprobs(m2, hid[cond], lab, r)
            lx, tgt_x = exec_logprobs(m2, hx, xlab, r)
            assert torch.equal(tgt, tgt_x)
            n = tgt.numel()
            idx = torch.arange(n, device="cuda")
            ce = {c: -lp[c].gather(1, tgt[:, None]).squeeze(1) for c in lp}
            ce["explicit_student"] = -lx.gather(1, tgt[:, None]).squeeze(1)
            for c, v in ce.items():
                A.add(f"ce_{c}", v.sum(), n)
                A.add(f"ce_{c}_nonrep", v[nonrep].sum(), int(nonrep.sum()))
            for other in ("zero", "shuffled"):
                kl = (lp["learned"].exp() * (lp["learned"] - lp[other])).sum(-1)
                A.add(f"kl_learned_vs_{other}", kl.sum(), n)
                A.add(f"kl_learned_vs_{other}_nonrep", kl[nonrep].sum(), int(nonrep.sum()))
                A.add(f"top1_agree_learned_vs_{other}", (lp["learned"].argmax(-1) == lp[other].argmax(-1)).sum(), n)
                for lo, hi in BUCKETS:
                    sel = (idx >= lo) & (idx < hi)
                    if sel.any():
                        A.add(f"kl_learned_vs_{other}_pos{lo}-{hi if hi < 10**9 else 'end'}", kl[sel].sum(), int(sel.sum()))
            klx = (lx.exp() * (lx - lp["learned"])).sum(-1)
            A.add("kl_explicit_vs_learned", klx.sum(), n)
            # Test C
            for name, hp in patched.items():
                lpp, _ = exec_logprobs(m2, hp, lab, r)
                cep = -lpp.gather(1, tgt[:, None]).squeeze(1)
                C.add(f"ce_patched_{name}", cep.sum(), n)
                C.add(f"ce_patched_{name}_nonrep", cep[nonrep].sum(), int(nonrep.sum()))
                C.add(f"kl_explicit_vs_patched_{name}", (lx.exp() * (lx - lpp)).sum(-1).sum(), n)
            # per-layer alignment
            for c in ("learned", "zero", "shuffled"):
                for l in LAYERS:
                    align[c].add(f"layer_{l}", norm_l1(bnd[c][l][r:r + 1], th[l][r:r + 1]).item())
            for l in LAYERS:
                align["explicit_student"].add(f"layer_{l}", norm_l1(bx[l][r:r + 1], th[l][r:r + 1]).item())
        print(f"A/C batch {s // args.batch_size} done", flush=True)
    a, c = A.out(), C.out()
    for name in patch_sets:
        for suf in ("", "_nonrep"):
            gap = a[f"ce_learned{suf}"] - a[f"ce_explicit_student{suf}"]
            c[f"transfer_closed_{name}{suf}"] = (a[f"ce_learned{suf}"] - c[f"ce_patched_{name}{suf}"]) / gap if gap else None
    al = {cond: acc.out() for cond, acc in align.items()}
    for cond in al:
        vals = [al[cond][f"layer_{l}"] for l in LAYERS]
        al[cond]["mean_all_layers"] = sum(vals) / len(vals)
    del m2
    torch.cuda.empty_cache()
    # Stage-1 reference for Test C: empty-plan path + teacher boundary patch
    m1 = load_stage2_model(args.stage1, latent_embed_builder_no_grad=True, latent_projector_dropout=0.0,
                           dtype=torch.bfloat16).cuda().eval()
    C1 = Acc()
    for s in range(0, len(rows_all), args.batch_size):
        chunk = rows_all[s: s + args.batch_size]
        b0 = collate([build_views(tok, r, 0, T, 4096) for r in chunk])
        b = collate([build_views(tok, r, args.latent_steps, T, 4096) for r in chunk])
        th = cache.get(b["record_ids"], LAYERS, "cuda")
        args0 = dict(input_ids=b0["student_input_ids"], mask=b0["student_attention_mask"], pos=b0["student_position_ids"],
                     capture_pos=b0["student_boundary"])
        h_empty, _ = run(m1, **args0)
        h_patch, _ = run(m1, **args0, patch={l: th[l] for l in LAYERS})
        hx, _ = run(m1, input_ids=b["explicit_input_ids"], mask=b["explicit_attention_mask"],
                    pos=b["explicit_position_ids"], capture_pos=b["explicit_boundary"])
        xlab = b["explicit_labels"].clone()
        xlab[~(b["explicit_seg_E"] | b["explicit_seg_A"] | b["explicit_seg_C"])] = -100
        for r in range(len(chunk)):
            nonrep = ~rep_mask(b0["student_seg_R"], b0["student_labels"], r)
            for name, h in (("empty_plan", h_empty), ("empty_plan_patched_all", h_patch)):
                l1, tgt = exec_logprobs(m1, h, b0["student_labels"], r)
                v = -l1.gather(1, tgt[:, None]).squeeze(1)
                C1.add(f"ce_{name}", v.sum(), tgt.numel())
                C1.add(f"ce_{name}_nonrep", v[nonrep].sum(), int(nonrep.sum()))
            lx, tgt = exec_logprobs(m1, hx, xlab, r)
            v = -lx.gather(1, tgt[:, None]).squeeze(1)
            C1.add("ce_gold_plan", v.sum(), tgt.numel())
            C1.add("ce_gold_plan_nonrep", v[nonrep].sum(), int(nonrep.sum()))
    c1 = C1.out()
    for suf in ("", "_nonrep"):
        gap = c1[f"ce_empty_plan{suf}"] - c1[f"ce_gold_plan{suf}"]
        c1[f"transfer_closed_all{suf}"] = (c1[f"ce_empty_plan{suf}"] - c1[f"ce_empty_plan_patched_all{suf}"]) / gap
    return {"test_a": a, "test_c_stage2": c, "test_c_stage1_reference": c1, "boundary_alignment_per_layer": al}


# ------------------------------------------------------------------ Test B (Stage-1 teacher)
STEP_RE = re.compile(r"^\s*(\d+)\.\s*(.*)$")


def split_plan(plan):
    lines = plan.split("\n")
    head = []
    steps = []
    for ln in lines:
        mm = STEP_RE.match(ln)
        if mm:
            steps.append(mm.group(2).rstrip())
        elif steps:
            return None, None            # non-step text after steps: keep it simple, skip record
        else:
            head.append(ln)
    return head, steps


def join_plan(head, steps):
    return "\n".join(head + [f"{i + 1}. {s}  " for i, s in enumerate(steps)])


def teacher_sequence(tok, q, plan, execution):
    prompt = tok.encode(render_prompt(tok, q), add_special_tokens=False)
    resp = PLAN_OPEN + plan + SEPARATOR + execution + EXEC_CLOSE + EOS
    enc = tok(resp, add_special_tokens=False, return_offsets_mapping=True)
    c0 = len(PLAN_OPEN) + len(plan) + len(SEPARATOR)
    es = next(i for i, (a, _) in enumerate(enc["offset_mapping"]) if a >= c0)
    ids = prompt + enc["input_ids"]
    b = len(prompt) + es - 1
    assert ids[b] == 397, tok.convert_ids_to_tokens(ids[b])
    labels = [-100] * (b + 1) + ids[b + 1:]
    return ids, labels, b


@torch.no_grad()
def test_b(args, tok, frame):
    m1 = load_stage2_model(args.stage1, latent_embed_builder_no_grad=True, latent_projector_dropout=0.0,
                           dtype=torch.bfloat16).cuda().eval()
    T = stage2_tokens(tok)
    rows = [frame.iloc[i].to_dict() for i in range(len(frame))]
    B, dist_l = Acc(), {v: Acc() for v in ("B_reversed", "M_other_record", "N_empty")}
    used = 0
    gen_items = []
    for i, r in enumerate(rows):
        head, steps = split_plan(r["plan"])
        if not steps or len(steps) < 3:
            continue
        other = rows[(i + 1) % len(rows)]["plan"]
        variants = {"A_gold": r["plan"], "B_reversed": join_plan(head, steps[::-1]),
                    "M_other_record": other, "N_empty": ""}
        lps, bnds, tgt0, nonrep = {}, {}, None, None
        for name, plan in variants.items():
            ids, labels, b = teacher_sequence(tok, r["question"], plan, r["execution"])
            t = torch.tensor([ids])
            hid, bnd = run(m1, input_ids=t, mask=torch.ones_like(t), pos=torch.arange(t.size(1)).unsqueeze(0),
                           capture_pos=torch.tensor([b]))
            lab = torch.tensor([labels])
            lps[name], tgt = exec_logprobs(m1, hid, lab, 0)
            bnds[name] = bnd
            if tgt0 is None:
                tgt0 = tgt
                v = build_views(tok, r, 1, T, 4096)
                nonrep = ~torch.tensor(v["explicit_rep"][v["explicit_boundary"] + 1:], device="cuda")
            assert torch.equal(tgt, tgt0), "execution tokens differ across plan variants"
        n = tgt0.numel()
        idx = torch.arange(n, device="cuda")
        for name, lp in lps.items():
            ce = -lp.gather(1, tgt0[:, None]).squeeze(1)
            B.add(f"ce_{name}", ce.sum(), n)
            B.add(f"ce_{name}_nonrep", ce[nonrep].sum(), int(nonrep.sum()))
            if name != "A_gold":
                kl = (lps["A_gold"].exp() * (lps["A_gold"] - lp)).sum(-1)
                B.add(f"kl_A_vs_{name}", kl.sum(), n)
                B.add(f"kl_A_vs_{name}_nonrep", kl[nonrep].sum(), int(nonrep.sum()))
                for lo, hi in BUCKETS:
                    sel = (idx >= lo) & (idx < hi)
                    if sel.any():
                        B.add(f"kl_A_vs_{name}_pos{lo}-{hi if hi < 10**9 else 'end'}", kl[sel].sum(), int(sel.sum()))
                for l in LAYERS:
                    dist_l[name].add(f"layer_{l}", norm_l1(bnds[name][l], bnds["A_gold"][l]).item())
        used += 1
        if len(gen_items) < args.n_gen:
            gen_items.append((r, head, steps))
        if used >= args.n:
            break
    res = {"n_records": used, "teacher_forced": B.out()}
    dl = {k: a.out() for k, a in dist_l.items()}
    for k in dl:
        vals = [dl[k][f"layer_{l}"] for l in LAYERS]
        dl[k]["mean_all_layers"] = sum(vals) / len(vals)
    res["teacher_boundary_distance_to_A_per_layer"] = dl
    # strategy switching: greedy generation under plan A vs reversed plan B
    eos = [tok.convert_tokens_to_ids("<|im_end|>")]
    sw = {"follows_given_first_step": {"A_gold": 0, "B_reversed": 0}, "follows_A_order_under_B": 0,
          "order_exact_match": {"A_gold": 0, "B_reversed": 0}, "n": len(gen_items)}
    samples = []
    for name in ("A_gold", "B_reversed"):
        prompts = []
        for r, head, steps in gen_items:
            plan = r["plan"] if name == "A_gold" else join_plan(head, steps[::-1])
            prompts.append(tok.encode(render_prompt(tok, r["question"]), add_special_tokens=False)
                           + tok.encode(PLAN_OPEN + plan + SEPARATOR, add_special_tokens=False))
        outs = []
        for s in range(0, len(prompts), 32):
            g = m1.generate_plan_latent(prompts[s:s + 32], 0, [], args.gen_tokens, eos, tok.pad_token_id)
            outs += g["token_ids"]
        for (r, head, steps), ids in zip(gen_items, outs):
            text = tok.decode(ids, skip_special_tokens=True)
            given = steps if name == "A_gold" else steps[::-1]
            seq = [mm.group(2).strip() for ln in text.split("\n") if (mm := STEP_RE.match(ln))]
            seq = [x for x in seq if any(x.startswith(st.strip()[:40]) for st in steps)]
            first = seq[0] if seq else ""
            match = lambda a, b_: a.startswith(b_.strip()[:40])
            if seq and match(first, given[0]):
                sw["follows_given_first_step"][name] += 1
            if name == "B_reversed" and seq and match(first, steps[0]):
                sw["follows_A_order_under_B"] += 1
            k = min(len(seq), len(given))
            if k >= 3 and all(match(seq[j], given[j]) for j in range(k)):
                sw["order_exact_match"][name] += 1
            if len(samples) < 6:
                samples.append({"record_id": r["record_id"], "plan_variant": name, "given_steps": given,
                                "generated_head": text[:800]})
    n = max(1, sw["n"])
    sw["rates"] = {"first_step_follows_given_A": sw["follows_given_first_step"]["A_gold"] / n,
                   "first_step_follows_given_B": sw["follows_given_first_step"]["B_reversed"] / n,
                   "under_B_first_step_is_A_first": sw["follows_A_order_under_B"] / n,
                   "order_exact_match_A": sw["order_exact_match"]["A_gold"] / n,
                   "order_exact_match_B": sw["order_exact_match"]["B_reversed"] / n}
    res["strategy_switching"] = sw
    res["generation_samples"] = samples
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["b", "ac"], required=True)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", default=None)
    ap.add_argument("--val", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--n_gen", type=int, default=64)
    ap.add_argument("--gen_tokens", type=int, default=512)
    ap.add_argument("--latent_steps", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = load_tokenizer(args.stage1)
    frame = pd.read_parquet(args.val)
    t0 = time.time()
    if args.test == "ac":
        cache = TeacherBoundaryCache(args.cache_dir, "val", {})
        res = test_ac(args, tok, frame.iloc[: args.n], cache)
    else:
        res = test_b(args, tok, frame.iloc[: max(args.n * 3, args.n)])   # extra rows: some plans are skipped
    res["meta"] = {"test": args.test, "stage1": args.stage1, "stage2": args.stage2, "val": args.val, "n": args.n,
                   "latent_steps": args.latent_steps, "seconds": round(time.time() - t0),
                   "decoding": f"greedy, max_new_tokens={args.gen_tokens}" if args.test == "b" else "teacher forcing"}
    dump_json(res, args.out)
    print("DIAG_DONE", args.test, json.dumps(res["meta"]), flush=True)


if __name__ == "__main__":
    main()
