"""Contract tests for the two-loss PlanRec Stage 2 (objective_mode=planrec_two_loss), real Stage-1 checkpoint + data.

Run: pytest -q later/src/train/test/test_stage2_planrec.py   (GPU required for the model tests)
"""
import os
import random

import pandas as pd
import pytest
import torch

from later.src.train.stage2_common import load_stage2_model, load_tokenizer
from later.src.train.stage2_plan_dataset import (EOS, EXEC_CLOSE, PLAN_HEADER, PLAN_OPEN, SEPARATOR, Stage2Collator,
                                                 build_views, stage2_tokens)
from later.src.train.stage2_planrec_monitor import CONDITIONS, PlanRecMonitor, build_latents, fixed_derangement
from later.src.train.stage2_planrec_objective import PlanRecModule, check_disabled

CKPT = os.environ.get("STAGE2_CKPT", "/scratch/hz33227/e2c/Explore-Execute-Chain/outputs/e2c_qwen3_4b_sft/best_hf")
DATA = os.environ.get("STAGE2_DATA_DIR", "/scratch/hz33227/e2c/stage2/data")
CFG_B = dict(lambda_p=0.5, execution_ce_weight=1.0)
L = 8


@pytest.fixture(scope="session")
def tok():
    return load_tokenizer(CKPT)


@pytest.fixture(scope="session")
def frames():
    return {s: pd.read_parquet(os.path.join(DATA, f"stage2_{s}.parquet")) for s in ("train", "val")}


@pytest.fixture(scope="session")
def model():
    m = load_stage2_model(CKPT, latent_embed_builder_no_grad=False, latent_projector_dropout=0.0,
                          dtype=torch.float32, latent_output_rms="token_embedding").cuda()
    m.eval()
    return m


def batch_of(tok, rows, L=L):
    T = stage2_tokens(tok)
    views = [build_views(tok, r, L, T, 4096) for r in rows]
    return Stage2Collator(tok.pad_token_id)(views), views


def zero_grads(model):
    for p in model.parameters():
        p.grad = None


# ------------------------------------------------------------------ 1. data / masks
def test_planrec_targets_and_masks(tok, frames):
    T = stage2_tokens(tok)
    n_open = len(T.plan_open_ids)
    rng = random.Random(0)
    for split in ("train", "val"):
        f = frames[split]
        for i in rng.sample(range(len(f)), 600):
            r = f.iloc[i].to_dict()
            assert PLAN_OPEN + r["plan"] + SEPARATOR + r["execution"] + EXEC_CLOSE == r["stage1_solution"]
            v = build_views(tok, r, L, T, 4096)
            ids, lab = v["planrec_ids"], v["planrec_labels"]
            assert len(ids) == len(lab) == len(v["planrec_content"]) == len(v["planrec_math"])
            assert ids[:L] == [T.pad_id] * L and ids[L:L + n_open] == T.plan_open_ids
            assert all(x == -100 for x in lab[:L + n_open])          # latent slots + fixed prefix never supervised
            tgt = lab[L + n_open:]
            assert tgt == ids[L + n_open:] and all(x != -100 for x in tgt)
            # targets are exactly the Stage-1 plan segment tokens: plan text + the separator's leading newline
            assert tok.decode(tgt) == r["plan"] + "\n"
            sp = v["explicit_seg"][v["prompt_len"]:]
            resp = v["explicit_ids"][v["prompt_len"]:]
            assert tgt == [t for t, s in zip(resp, sp) if s == "P"]
            # execution CE view unchanged: E + A + closing only
            s_t = [x for x in v["student_labels"] if x != -100]
            assert tok.decode(s_t) == r["execution"] + EXEC_CLOSE + EOS
            # no question / execution text inside the PlanRec context
            ctx = tok.decode(ids[L:])
            assert ctx == PLAN_OPEN + r["plan"] + "\n"
            assert r["execution"][:200] not in ctx or r["execution"][:200] in r["plan"]


def test_content_mask_excludes_template(tok, frames):
    T = stage2_tokens(tok)
    r = frames["val"].iloc[0].to_dict()
    v = build_views(tok, r, L, T, 4096)
    toks = v["planrec_ids"]
    cont = v["planrec_content"]
    head_ids = tok.encode(PLAN_OPEN + PLAN_HEADER, add_special_tokens=False)
    start = L + len(T.plan_open_ids)
    n_head = len(head_ids) - len(T.plan_open_ids)
    assert not any(cont[start: start + n_head])                 # header line is template
    text = [tok.decode([t]) for t in toks]
    for k in range(start, len(toks)):
        if text[k].strip() in ("1.", "2.", "3.", "") :
            assert not cont[k], (k, text[k])
    assert sum(cont) > 20 and all(c or not m for c, m in zip(cont, v["planrec_math"]))  # math subset of content


# ------------------------------------------------------------------ 2. leakage / causality
@pytest.mark.parametrize("field", ["execution", "plan"])
def test_latents_ignore_plan_execution_answer(model, tok, frames, field):
    r = frames["val"].iloc[3].to_dict()
    r2 = dict(r)
    if field == "execution":
        r2["execution"] = "Totally different execution. The answer is $\\boxed{12345}$."
        r2["answer_char_start_in_execution"] = r2["execution"].index("\\boxed")
    else:
        r2["plan"] = "I need to make a step-by-step plan.\n1. Do something unrelated."
    r2["stage1_solution"] = PLAN_OPEN + r2["plan"] + SEPARATOR + r2["execution"] + EXEC_CLOSE
    b1, _ = batch_of(tok, [r])
    b2, _ = batch_of(tok, [r2])
    z1, z2 = build_latents(model, b1), build_latents(model, b2)
    assert torch.equal(z1, z2)
    r3 = dict(r)
    r3["question"] = r["question"] + " Also, simplify."
    assert not torch.equal(z1, build_latents(model, batch_of(tok, [r3])[0]))


def test_planrec_causal_and_shift(model, tok, frames):
    b, v = batch_of(tok, [frames["val"].iloc[5].to_dict()])
    dev = "cuda"
    z = torch.randn(1, L, model.config.hidden_size, device=dev) * 0.02
    ids = b["planrec_input_ids"].to(dev)
    with torch.no_grad():
        o = model.forward_plan_reconstruction(z, ids, b["planrec_attention_mask"].to(dev), b["planrec_labels"].to(dev),
                                              return_logits=True)
        first = L + len(stage2_tokens(tok).plan_open_ids)           # index of the first plan token
        n_t = ids.size(1) - first
        assert o["token_nll"].numel() == n_t
        # manual first/last target: logits at position first-1 predict ids[first]; last target = last plan token
        x = torch.cat([z, model.get_input_embeddings()(ids)[:, L:]], dim=1)
        h = model.model(inputs_embeds=x, position_ids=torch.arange(ids.size(1), device=dev)[None]).last_hidden_state
        lg = model.lm_head(h).float()
        ce = torch.nn.functional.cross_entropy(lg[0, first - 1: -1], ids[0, first:], reduction="none")
        assert torch.allclose(ce, o["token_nll"], atol=1e-4)
        # future plan tokens do not leak: changing token k changes nothing before k
        k = first + n_t // 2
        ids2 = ids.clone(); ids2[0, k] = (ids2[0, k] + 7) % 1000
        lab2 = b["planrec_labels"].clone().to(dev); lab2[0, k] = ids2[0, k]
        o2 = model.forward_plan_reconstruction(z, ids2, b["planrec_attention_mask"].to(dev), lab2)
        j = k - first                                               # target index of token k
        assert torch.allclose(o["token_nll"][:j], o2["token_nll"][:j], atol=1e-5)
        assert not torch.allclose(o["token_nll"][j + 1:], o2["token_nll"][j + 1:], atol=1e-5)


# ------------------------------------------------------------------ 3. gradients
def test_planrec_only_gradients_reach_z1_projector_backbone(model, tok, frames):
    b, _ = batch_of(tok, [frames["val"].iloc[i].to_dict() for i in (0, 1)])
    mod = PlanRecModule(model, CFG_B)
    zero_grads(model)
    out = mod(b, debug_latent_grad=True)
    z1 = out["latent"]["projector_outputs"][0]
    out["l_planrec"].backward()
    pg = [p.grad for p in model.latent_projector.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in pg) and sum(float(g.norm()) for g in pg) > 0
    assert z1.grad is not None and float(z1.grad.norm()) > 0
    bb = model.model.layers[0].self_attn.q_proj.weight.grad
    assert bb is not None and float(bb.norm()) > 0 and torch.isfinite(bb).all()
    zero_grads(model)
    out = mod(b, debug_latent_grad=True)
    z1 = out["latent"]["projector_outputs"][0]
    out["l_exec"].backward()
    assert float(z1.grad.norm()) > 0
    assert all(float(p.grad.norm()) > 0 for p in model.latent_projector.parameters())
    zero_grads(model)


def test_total_is_exactly_two_terms_and_control_skips_planrec(model, tok, frames):
    b, _ = batch_of(tok, [frames["val"].iloc[2].to_dict()])
    with torch.no_grad():
        o = PlanRecModule(model, CFG_B)(b)
        assert torch.allclose(o["loss"], o["l_exec"] + 0.5 * o["l_planrec"])
        oa = PlanRecModule(model, dict(lambda_p=0.0, execution_ce_weight=1.0))(b)
        assert "planrec" not in oa and float(oa["l_planrec"]) == 0.0
        assert torch.allclose(oa["loss"], o["l_exec"])
    for bad in (dict(lambda_a=0.1), dict(lambda_r=0.3), dict(execution_kl_enabled=True), dict(noise_std=0.1),
                dict(rl_enabled=True), dict(halting_loss_enabled=True), dict(teacher_kl_enabled=True)):
        with pytest.raises(ValueError):
            check_disabled(bad)


def test_z_in_planrec_is_the_inserted_learned_latent(model, tok, frames):
    b, _ = batch_of(tok, [frames["val"].iloc[i].to_dict() for i in (0, 1)])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):   # same precision as training / probes
        o = PlanRecModule(model, CFG_B)(b)
        z = o["z"]
        rows = torch.arange(2, device="cuda").unsqueeze(1)
        assert torch.equal(o["latent"]["inputs_embeds"][rows, b["latent_positions"].cuda()], z.to(o["latent"]["inputs_embeds"].dtype))
        assert torch.allclose(z.float(), build_latents(model, b), atol=1e-5)
        # not the placeholder embeddings
        ph = model.get_input_embeddings()(b["student_input_ids"].cuda())[rows, b["latent_positions"].cuda()]
        assert not torch.allclose(z, ph)


# ------------------------------------------------------------------ 4. ablations
def test_override_matches_builder_and_modes(model, tok, frames):
    b, _ = batch_of(tok, [frames["val"].iloc[i].to_dict() for i in (0, 1, 2)])
    dev = "cuda"
    kw = dict(input_ids=b["student_input_ids"].to(dev), attention_mask=b["student_attention_mask"].to(dev),
              position_ids=b["student_position_ids"].to(dev), labels=b["student_labels"].to(dev),
              boundary_positions=b["student_boundary"].to(dev), layer_ids=[36],
              latent_positions=b["latent_positions"].to(dev), latent_start_positions=b["latent_start_positions"].to(dev))
    with torch.no_grad():
        ref = model.forward_plan_distill_stage2(**kw)
        z = ref["latent_embeddings"]
        ov = model.forward_plan_distill_stage2(**kw, latent_override=z)
        assert ov["builder_stats"].get("latent_override") == 1 and "latent_calls" not in ov["builder_stats"]
        assert torch.allclose(ref["token_nll"], ov["token_nll"], atol=1e-4)
        zr = model.forward_plan_distill_stage2(**kw, latent_mode="zero")
        zo = model.forward_plan_distill_stage2(**kw, latent_override=torch.zeros_like(z))
        assert torch.allclose(zr["token_nll"], zo["token_nll"], atol=1e-4)
        sr = model.forward_plan_distill_stage2(**kw, latent_mode="shuffled")
        so = model.forward_plan_distill_stage2(**kw, latent_override=torch.roll(z, -1, 0))
        assert torch.allclose(sr["token_nll"], so["token_nll"], atol=1e-4)
        assert not torch.allclose(ref["token_nll"], so["token_nll"], atol=1e-4)


def test_monitor_conditions_read_only(model, tok, frames):
    mon = PlanRecMonitor(model, tok, frames["val"], L, probe_n=16, probe_batch_size=8, grad_batch_size=2)
    before = [p.detach().clone() for p in model.latent_projector.parameters()]
    zero_grads(model)
    res = mon.probe(return_records=True)
    g = mon.grad_split()
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(a, p) for a, p in zip(before, model.latent_projector.parameters()))
    recs = res["_records"]
    assert len(recs) == 16 and all(set(r) == set(CONDITIONS) for r in recs.values())
    for r in recs.values():   # identical targets for every condition
        assert len({(r[c]["exec_n"], r[c]["rec_n"], r[c]["rec_content_n"]) for c in CONDITIONS}) == 1
    perm = fixed_derangement(16)
    assert all(perm[i] != i for i in range(16)) and perm == fixed_derangement(16)
    Z = mon.last_Z
    assert torch.allclose(mon._cond_latents(Z, "mean", [0, 1])[0], Z.mean(0))
    assert torch.equal(mon._cond_latents(Z, "shuffled", [0])[0], Z[perm[0]])
    assert g["grad/projector_planrec_unweighted"] > 0 and g["grad/z1_exec"] > 0
    assert 0.5 < res["val/latent_rms_ratio"] < 2.0
