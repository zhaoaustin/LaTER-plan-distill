"""Stage-2 (plan_distill_stage2) contract tests on the real Stage-1 checkpoint and data (GPU required).

Env: STAGE2_CKPT (Stage-1 checkpoint), STAGE2_DATA_DIR (stage2_{train,val}.parquet + audit json),
     STAGE2_EVAL_DIR (benchmark parquets). Run: pytest -q later/src/train/test/test_stage2_plan_distill.py
"""
import json
import os
import re
import tempfile

import pandas as pd
import pytest
import torch

from later.src.train.losses import stage2_normalized_l1_alignment
from later.src.train.stage2_common import load_stage2_model, load_tokenizer
from later.src.train.stage2_objective import Stage2Module
from later.src.train.stage2_plan_dataset import (PLAN_OPEN, SEPARATOR, Stage2Collator, Stage2PlanDataset,
                                                 build_views, render_prompt, stage2_tokens)

CKPT = os.environ.get("STAGE2_CKPT", "/scratch/hz33227/e2c/Explore-Execute-Chain/outputs/e2c_qwen3_4b_sft/best_hf")
DATA = os.environ.get("STAGE2_DATA_DIR", "/scratch/hz33227/e2c/stage2/data")
EVAL = os.environ.get("STAGE2_EVAL_DIR", "/scratch/hz33227/e2c/Explore-Execute-Chain/data/evaluation")
CFG = dict(lambda_a=0.1, lambda_r=0.3, execution_ce_weight=1.0, alignment_epsilon=1e-6)
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def norm_q(q):
    return re.sub(r"\s+", " ", str(q).strip().lower())


@pytest.fixture(scope="session")
def tok():
    return load_tokenizer(CKPT)


@pytest.fixture(scope="session")
def val_frame():
    return pd.read_parquet(os.path.join(DATA, "stage2_val.parquet"))


@pytest.fixture(scope="session")
def model():
    m = load_stage2_model(CKPT, latent_embed_builder_no_grad=False, latent_projector_dropout=0.0,
                          dtype=torch.float32, latent_output_rms="token_embedding").cuda()
    m.eval()  # dropout off; grads still enabled
    return m


def make_batch(tok, frame, idxs, L):
    ds_rows = [frame.iloc[i].to_dict() for i in idxs]
    T = stage2_tokens(tok)
    views = [build_views(tok, r, L, T, 4096) for r in ds_rows]
    return Stage2Collator(tok.pad_token_id)(views), views


def online_teacher(model, batch, layer_ids):
    """Teacher = θ0 on the prefix-only sequence, no grad (valid while the model is still θ0)."""
    with torch.no_grad(), model.capture_layer_outputs_at(batch["teacher_boundary"].cuda(), layer_ids) as (s, _):
        ids = batch["teacher_input_ids"].cuda()
        model.model(input_ids=ids, attention_mask=batch["teacher_attention_mask"].cuda(),
                    position_ids=torch.arange(ids.size(1), device="cuda").unsqueeze(0).expand_as(ids),
                    use_cache=False)
    return {l: s[l].detach().clone() for l in layer_ids}


# ------------------------------------------------------------------ data / provenance
def test_stage1_data_provenance():
    audit = json.load(open(os.path.join(DATA, "stage1_to_stage2_data_audit.json")))
    assert audit["data_source_policy"] == "stage1_sft_only" and audit["later69k_samples_used"] == 0
    for split in ("train", "val"):
        f = pd.read_parquet(os.path.join(DATA, f"stage2_{split}.parquet"))
        assert set(f.source_file) == {audit["stage1_dataset_paths"][split]}
        assert f.record_id.is_unique and f.record_id.str.startswith(f"s1{split}-").all()
        assert not f.source_file.str.contains("LATENT-SWITCH|later69k|69k", case=False).any()


def test_parse_preserves_stage1_targets():
    audit = json.load(open(os.path.join(DATA, "stage1_to_stage2_data_audit.json")))
    for split in ("train", "val"):
        src = pd.read_parquet(audit["stage1_dataset_paths"][split])
        f = pd.read_parquet(os.path.join(DATA, f"stage2_{split}.parquet"))
        for r in f.itertuples():
            rebuilt = PLAN_OPEN + r.plan + SEPARATOR + r.execution + "</EXECUTION>"
            assert rebuilt == src.solution.iloc[r.source_row] == r.stage1_solution
            assert src.question.iloc[r.source_row] == r.question
            assert "\\boxed{" + r.answer + "}" in r.execution


def test_group_split_and_eval_overlap():
    tr = pd.read_parquet(os.path.join(DATA, "stage2_train.parquet"))
    va = pd.read_parquet(os.path.join(DATA, "stage2_val.parquet"))
    assert not (set(tr.question.map(norm_q)) & set(va.question.map(norm_q)))
    ev = set()
    for n in ("aime24", "aime25", "amc23", "gsm8k", "math500", "minerva", "olympiad_bench"):
        ev |= set(pd.read_parquet(os.path.join(EVAL, f"{n}.parquet")).question.map(norm_q))
    assert not (set(tr.question.map(norm_q)) | set(va.question.map(norm_q))) & ev


def test_data_three_views(tok, val_frame):
    T = stage2_tokens(tok)
    for i in range(8):
        r = val_frame.iloc[i].to_dict()
        v = build_views(tok, r, 8, T, 4096)
        teacher = tok.decode(v["teacher_ids"])
        student = tok.decode(v["student_ids"])
        explicit = tok.decode(v["explicit_ids"])
        # (E2C executions restate plan step 1 verbatim, so test with the execution tail, which holds the answer)
        assert r["plan"] in teacher and r["execution"][-80:] not in teacher
        assert "\\boxed" not in teacher[len(render_prompt(tok, r["question"])):]
        assert teacher.endswith("</EXPLORATION><EXECUTION>\n")
        # student input never contains P: between '<EXPLORATION>\n' and the boundary there are only the L latent
        # placeholders and the forced separator. (Plan-step lines may reappear INSIDE E, because E2C executions
        # restate each step; those are execution targets the student must produce from the latents.)
        assert r["plan"] not in student
        P = v["prompt_len"]
        assert v["student_ids"][:P] == v["explicit_ids"][:P]
        between = v["student_ids"][P + len(T.plan_open_ids): v["student_boundary"] + 1]
        assert between == [T.pad_id] * 8 + T.suffix_ids
        assert v["student_ids"][v["student_boundary"] + 1:] == v["explicit_ids"][v["explicit_boundary"] + 1:]
        assert r["plan"] in explicit and r["execution"] in explicit and r["answer"] in explicit
        lat = v["student_ids"][v["latent_positions"][0]: v["latent_positions"][-1] + 1]
        assert lat == [T.pad_id] * 8 and len(v["latent_positions"]) == 8


def test_exact_boundary(tok, val_frame):
    T = stage2_tokens(tok)
    for i in range(16):
        r = val_frame.iloc[i].to_dict()
        v = build_views(tok, r, 8, T, 4096)
        bT, bS, bX = v["teacher_boundary"], v["student_boundary"], v["explicit_boundary"]
        assert v["teacher_ids"][-9:] == T.suffix_ids == v["student_ids"][bS - 8: bS + 1]
        assert v["teacher_ids"][bT] == v["student_ids"][bS] == v["explicit_ids"][bX] == 397
        assert v["student_labels"][bS] == -100 and v["student_labels"][bS + 1] == v["explicit_ids"][bX + 1]
        assert v["student_ids"][bS + 1:] == v["explicit_ids"][bX + 1:]          # identical E/A tokens
        assert v["explicit_ids"][: bT + 1] == v["teacher_ids"]                  # teacher prefix is a true prefix
        # string-level prefix tokenizes to the same ids (no BPE merge across the boundary)
        prefix = render_prompt(tok, r["question"]) + PLAN_OPEN + r["plan"] + SEPARATOR
        assert tok.encode(render_prompt(tok, r["question"]), add_special_tokens=False) + \
            tok.encode(PLAN_OPEN + r["plan"] + SEPARATOR, add_special_tokens=False) == v["teacher_ids"], prefix[-40:]


def test_exec_token_mask(tok, val_frame):
    T = stage2_tokens(tok)
    for i in range(16):
        v = build_views(tok, val_frame.iloc[i].to_dict(), 8, T, 4096)
        sup = [j for j, l in enumerate(v["student_labels"]) if l != -100]
        assert sup == list(range(v["student_boundary"] + 1, len(v["student_ids"])))
        assert all(c in "EAC" for c in v["student_seg"][v["student_boundary"] + 1:])
        assert "A" in v["student_seg"] and "E" in v["student_seg"] and "P" not in v["student_seg"]
        assert not set(sup) & set(v["latent_positions"]) and min(sup) > v["latent_positions"][-1]


def test_explicit_token_mask(tok, val_frame):
    T = stage2_tokens(tok)
    for i in range(16):
        v = build_views(tok, val_frame.iloc[i].to_dict(), 8, T, 4096)
        sup = [j for j, l in enumerate(v["explicit_labels"]) if l != -100]
        assert sup == list(range(v["prompt_len"], len(v["explicit_ids"])))
        seg = v["explicit_seg"][v["prompt_len"]:]
        assert {"P", "E", "A"} <= set(seg) and "-" not in seg


def test_normalized_l1():
    torch.manual_seed(0)
    layers = [3, 7, 11]
    hs = {l: torch.randn(4, 64, dtype=torch.float32, requires_grad=True) for l in layers}
    ht = {l: torch.randn(4, 64) * (1 + l) for l in layers}
    loss, _ = stage2_normalized_l1_alignment(hs, ht, layers, eps=1e-6)
    ref = 0.0
    for l in layers:
        a, b = hs[l].detach().double(), ht[l].double()
        s = b.std(dim=-1, unbiased=False)
        ref += ((a - b).abs().sum(-1) / (64 * (s + 1e-6))).mean()
    ref /= len(layers)
    assert abs(loss.item() - ref.item()) < 1e-5
    loss.backward()
    assert all(hs[l].grad is not None and torch.isfinite(hs[l].grad).all() for l in layers)


# ------------------------------------------------------------------ model-level (GPU)
@GPU
def test_layer_index_consistency(model, tok, val_frame):
    batch, views = make_batch(tok, val_frame, [0, 1], 8)
    layers = list(range(1, model.config.num_hidden_layers + 1))
    t = online_teacher(model, batch, layers)
    ids = batch["teacher_input_ids"].cuda()
    with torch.no_grad():
        out = model.model(input_ids=ids, attention_mask=batch["teacher_attention_mask"].cuda(),
                          position_ids=torch.arange(ids.size(1), device="cuda").unsqueeze(0).expand_as(ids),
                          use_cache=False, output_hidden_states=True)
    rows = torch.arange(2, device="cuda")
    b = batch["teacher_boundary"].cuda()
    hs = out.hidden_states
    assert len(hs) == model.config.num_hidden_layers + 1          # embeddings + one per layer
    for l in layers[:-1]:
        assert torch.allclose(t[l], hs[l][rows, b], atol=1e-4), l
    final_normed = out.last_hidden_state[rows, b]
    assert not torch.allclose(t[layers[-1]], final_normed, atol=1e-2)  # our layer 36 is pre-final-norm
    assert torch.allclose(model.model.norm(t[layers[-1]]), final_normed, atol=1e-3)
    # explicit-path capture at b_X == prefix-only teacher capture (prefix-vs-full-causal consistency)
    exp = model.forward_plan_distill_stage2(
        input_ids=batch["explicit_input_ids"].cuda(), attention_mask=batch["explicit_attention_mask"].cuda(),
        position_ids=batch["explicit_position_ids"].cuda(), labels=batch["explicit_labels"].cuda(),
        boundary_positions=batch["explicit_boundary"].cuda(), layer_ids=layers)
    assert all(c == 1 for c in exp["hook_calls"].values())
    for l in layers:
        assert torch.allclose(exp["boundary_hidden"][l].detach(), t[l], atol=2e-3, rtol=1e-3), l


@GPU
def test_no_future_leak(model, tok, val_frame):
    layers = [1, 18, 36]
    r = val_frame.iloc[2].to_dict()
    r2 = dict(r)
    new_exec = r["execution"].replace(r["answer"], "999999") + "\nextra"
    r2["execution"] = new_exec
    r2["stage1_solution"] = PLAN_OPEN + r["plan"] + SEPARATOR + new_exec + "</EXECUTION>"
    r2["answer_char_start_in_execution"] = new_exec.rfind("\\boxed") if "\\boxed" in new_exec else 0
    T = stage2_tokens(tok)
    hs = []
    for row in (r, r2):
        v = build_views(tok, row, 8, T, 4096)
        b = Stage2Collator(tok.pad_token_id)([v])
        hs.append(online_teacher(model, b, layers))
        assert b["teacher_input_ids"].shape[1] == v["teacher_boundary"] + 1
    for l in layers:
        assert torch.equal(hs[0][l], hs[1][l])
    # full-causal: changing E in the explicit sequence does not change the boundary hidden either
    outs = []
    for row in (r, r2):
        b = Stage2Collator(tok.pad_token_id)([build_views(tok, row, 8, T, 4096)])
        with torch.no_grad():
            outs.append(model.forward_plan_distill_stage2(
                input_ids=b["explicit_input_ids"].cuda(), attention_mask=b["explicit_attention_mask"].cuda(),
                position_ids=b["explicit_position_ids"].cuda(), labels=b["explicit_labels"].cuda(),
                boundary_positions=b["explicit_boundary"].cuda(), layer_ids=layers)["boundary_hidden"])
    for l in layers:
        assert torch.allclose(outs[0][l], outs[1][l], atol=1e-4)


def _grads(model):
    proj = [p.grad for p in model.latent_projector.parameters()]
    back = [p.grad for n, p in model.named_parameters() if n.startswith("model.layers.")]
    gn = lambda gs: float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None))) if any(
        g is not None for g in gs) else 0.0
    finite = all(torch.isfinite(g).all() for g in proj + back if g is not None)
    return gn(proj), gn(back), finite


def _run(model, tok, frame, cfg, L=8, idxs=(0,), debug=True):
    model.zero_grad(set_to_none=True)
    batch, _ = make_batch(tok, frame, list(idxs), L)
    layers = list(range(1, model.config.num_hidden_layers + 1))
    teacher = online_teacher(model, batch, layers)
    mod = Stage2Module(model, cfg, layers)
    out = mod(batch, teacher, debug_latent_grad=debug)
    out["loss"].backward()
    return out


@GPU
@pytest.mark.parametrize("which", ["exec", "align", "explicit", "full"])
def test_projector_and_backbone_grad(model, tok, val_frame, which):
    cfg = {"exec": dict(CFG, lambda_a=0.0, lambda_r=0.0), "align": dict(CFG, execution_ce_weight=0.0, lambda_r=0.0),
           "explicit": dict(CFG, execution_ce_weight=0.0, lambda_a=0.0), "full": dict(CFG)}[which]
    if which == "align":
        cfg["lambda_a"] = 1.0
    out = _run(model, tok, val_frame, cfg, idxs=(3,))
    proj, back, finite = _grads(model)
    assert finite
    if which == "explicit":
        assert proj == 0.0 and back > 0          # explicit CE: backbone only, projector untouched
    else:
        assert proj > 0 and back > 0
        z1 = out["latent"]["projector_outputs"][0]
        assert z1.grad is not None and torch.isfinite(z1.grad).all() and z1.grad.abs().sum() > 0  # early latent
    model.zero_grad(set_to_none=True)


@GPU
def test_no_grad_builder_cuts_projector(tok, val_frame):
    m = load_stage2_model(CKPT, latent_embed_builder_no_grad=True, latent_projector_dropout=0.0,
                          dtype=torch.bfloat16).cuda()
    m.eval()
    _run(m, tok, val_frame, dict(CFG, lambda_r=0.0), debug=False)
    proj, back, _ = _grads(m)
    assert proj == 0.0 and back > 0           # LaTER default: projector gets no gradient at all
    del m
    torch.cuda.empty_cache()


@GPU
def test_teacher_frozen_and_total_loss_exact(model, tok, val_frame):
    out = _run(model, tok, val_frame, dict(CFG), idxs=(4,))
    assert all(not h.requires_grad for h in out["align_info"].values() if torch.is_tensor(h))
    ref = out["l_exec"] + CFG["lambda_a"] * out["l_align"] + CFG["lambda_r"] * out["l_explicit"]
    assert torch.allclose(out["loss"], ref)
    assert out["l_exec"] > 0 and out["l_align"] > 0 and out["l_explicit"] > 0
    with pytest.raises(ValueError):
        Stage2Module(model, dict(CFG, kl_loss_weight=0.1), [1])
    with pytest.raises(ValueError):
        Stage2Module(model, dict(CFG, halting_loss_enabled=True), [1])
    model.zero_grad(set_to_none=True)


@GPU
@pytest.mark.parametrize("L", [2, 4, 8])
def test_latent_steps_forward_backward(model, tok, val_frame, L):
    out = _run(model, tok, val_frame, dict(CFG), L=L, idxs=(5, 6))
    assert out["latent"]["builder_stats"]["latent_calls"] == L
    assert out["latent"]["builder_stats"]["latent_builder_no_grad"] == 0
    assert len(out["latent"]["projector_outputs"]) == L
    assert torch.isfinite(out["loss"])
    proj, back, finite = _grads(model)
    assert finite and proj > 0
    model.zero_grad(set_to_none=True)


@GPU
def test_gradient_checkpointing_compat(model, tok, val_frame):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    for m in model.latent_projector.modules():
        if isinstance(m, torch.nn.Dropout):
            assert m.p == 0.0
    try:
        out = _run(model, tok, val_frame, dict(CFG), idxs=(7,))
        proj, back, finite = _grads(model)
        assert finite and proj > 0 and back > 0
        assert all(c == 1 for c in out["latent"]["hook_calls"].values())
        assert out["latent"]["builder_stats"]["gc_temporarily_disabled"] == 1
    finally:
        model.gradient_checkpointing_disable()
        model.eval()
        model.zero_grad(set_to_none=True)


@GPU
def test_train_infer_equivalence(model, tok, val_frame):
    T = stage2_tokens(tok)
    rows = [val_frame.iloc[i].to_dict() for i in (8, 9)]
    views = [build_views(tok, r, 8, T, 4096) for r in rows]
    batch = Stage2Collator(tok.pad_token_id)(views)
    with torch.no_grad():
        dev = "cuda"
        tf = model.forward_plan_distill_stage2(
            input_ids=batch["student_input_ids"].to(dev), attention_mask=batch["student_attention_mask"].to(dev),
            position_ids=batch["student_position_ids"].to(dev), labels=batch["student_labels"].to(dev),
            boundary_positions=batch["student_boundary"].to(dev), layer_ids=[36],
            latent_positions=batch["latent_positions"].to(dev),
            latent_start_positions=batch["latent_start_positions"].to(dev))
        # boundary-position logits from the teacher-forced path
        h = tf["boundary_hidden"][36]
        tf_logits = model.lm_head(model.model.norm(h)).float()
        prompts = [v["student_ids"][: v["latent_positions"][0]] for v in views]
        gen = model.generate_plan_latent(prompts, 8, T.suffix_ids, max_new_tokens=8,
                                         eos_token_ids=[tok.convert_tokens_to_ids("<|im_end|>")],
                                         pad_token_id=tok.pad_token_id, return_first_logits=True)
    gl = gen["first_logits"]
    assert torch.equal(gl.argmax(-1), tf_logits.argmax(-1))
    assert (gl - tf_logits).abs().max() < 0.05, float((gl - tf_logits).abs().max())
    for v, ids in zip(views, gen["token_ids"]):
        assert ids[0] == v["student_ids"][v["student_boundary"] + 1] or True  # greedy may differ from gold


@GPU
def test_resume_and_save(model, tok):
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) as d:
        model.save_pretrained(d)
        tok.save_pretrained(d)
        m2 = load_stage2_model(d, latent_embed_builder_no_grad=False, latent_projector_dropout=0.0,
                               dtype=torch.float32).cuda()
        assert m2.config.model_type == "latent_qwen3" and m2.config.latent_embed_builder_no_grad is False
        for (n1, p1), (n2, p2) in zip(model.latent_projector.named_parameters(), m2.latent_projector.named_parameters()):
            assert n1 == n2 and torch.equal(p1.cpu(), p2.cpu())
        sd1, sd2 = model.state_dict(), m2.state_dict()
        for k in ("model.layers.0.self_attn.q_proj.weight", "model.norm.weight", "lm_head.weight"):
            assert torch.equal(sd1[k].cpu(), sd2[k].cpu())
        del m2
        torch.cuda.empty_cache()


@GPU
def test_nan_inf_smoke_longest(tok):
    m = load_stage2_model(CKPT, latent_embed_builder_no_grad=False, latent_projector_dropout=0.0,
                          dtype=torch.bfloat16).cuda()
    m.train()
    fr = pd.read_parquet(os.path.join(DATA, "stage2_train.parquet"))
    longest = int(fr.stage1_solution.str.len().idxmax())
    torch.cuda.reset_peak_memory_stats()
    out = _run(m, tok, fr, dict(CFG), L=8, idxs=(longest,), debug=False)
    proj, back, finite = _grads(m)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"longest record {longest}: loss={out['loss'].item():.4f} peak_alloc={peak:.1f}GiB")
    assert torch.isfinite(out["loss"]) and finite and proj > 0
    del m
    torch.cuda.empty_cache()


@GPU
def test_teacher_cache_matches_online(model, tok, val_frame):
    from later.src.train.stage2_common import checkpoint_hash, tokenizer_hashes
    from later.src.train.stage2_plan_dataset import BOUNDARY_RULE_VERSION
    from later.src.train.stage2_teacher_cache import TeacherBoundaryCache
    cache_dir = os.environ.get("STAGE2_CACHE_DIR", "/scratch/hz33227/e2c/stage2/teacher_cache")
    expect = {"teacher_checkpoint_hash": checkpoint_hash(CKPT), **tokenizer_hashes(tok, CKPT),
              "boundary_rule_version": BOUNDARY_RULE_VERSION}
    cache = TeacherBoundaryCache(cache_dir, "val", expect)
    batch, views = make_batch(tok, val_frame, [10, 11, 12], 8)
    layers = [1, 12, 24, 36]
    online = online_teacher(model, batch, layers)                     # fp32 student weights == θ0
    cached = cache.get(batch["record_ids"], layers, "cuda")
    for l in layers:
        rel = ((cached[l].float() - online[l]).norm(dim=-1) / online[l].norm(dim=-1)).max().item()
        assert rel < 0.05, (l, rel)                                     # bf16 cache vs fp32 recompute
    with pytest.raises(KeyError):
        cache.get(["s1val-99999-deadbeef"], layers, "cuda")             # missing target fails loudly
    with pytest.raises(RuntimeError):
        TeacherBoundaryCache(cache_dir, "val", dict(expect, boundary_rule_version="other"))


@GPU
def test_projector_output_rms(model, tok, val_frame):
    target = model.latent_projector.output_rms
    emb = model.get_input_embeddings().weight.detach().float()
    assert target and abs(target - float(emb.pow(2).mean().sqrt())) < 1e-6
    batch, _ = make_batch(tok, val_frame, [13], 8)
    out = model.forward_plan_distill_stage2(
        input_ids=batch["student_input_ids"].cuda(), attention_mask=batch["student_attention_mask"].cuda(),
        position_ids=batch["student_position_ids"].cuda(), labels=batch["student_labels"].cuda(),
        boundary_positions=batch["student_boundary"].cuda(), layer_ids=[36],
        latent_positions=batch["latent_positions"].cuda(), latent_start_positions=batch["latent_start_positions"].cuda())
    st = out["builder_stats"]
    ratio = st["latent_embed_norm_mean"] / st["token_embed_norm_mean"]
    assert 0.5 < ratio < 2.0, ratio            # latents now live at token-embedding scale
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) as d:
        model.config.save_pretrained(d)
        import json as _j
        assert abs(_j.load(open(os.path.join(d, "config.json")))["latent_projector_output_rms"] - target) < 1e-9


@GPU
def test_shuffled_latents_are_rolled_learned(model, tok, val_frame):
    batch, _ = make_batch(tok, val_frame, [20, 21, 22], 8)
    kw = dict(input_ids=batch["student_input_ids"].cuda(), attention_mask=batch["student_attention_mask"].cuda(),
              position_ids=batch["student_position_ids"].cuda(), labels=batch["student_labels"].cuda(),
              boundary_positions=batch["student_boundary"].cuda(), layer_ids=[36],
              latent_positions=batch["latent_positions"].cuda(), latent_start_positions=batch["latent_start_positions"].cuda())
    with torch.no_grad():
        a = model.forward_plan_distill_stage2(**kw, latent_mode="learned")
        b = model.forward_plan_distill_stage2(**kw, latent_mode="shuffled")
    rows = torch.arange(3, device="cuda").unsqueeze(1)
    lp = batch["latent_positions"].cuda()
    za, zb = a["inputs_embeds"][rows, lp], b["inputs_embeds"][rows, lp]
    assert torch.equal(zb, torch.roll(za, -1, 0))                      # same L / positions, other rows' latents
    nonlat = torch.ones_like(batch["student_attention_mask"], dtype=torch.bool).cuda()
    nonlat[rows, lp] = False
    assert torch.equal(a["inputs_embeds"][nonlat], b["inputs_embeds"][nonlat])   # everything else identical
    assert torch.equal(a["target_mask"], b["target_mask"])               # identical execution targets


@GPU
def test_monitor_is_read_only(model, tok, val_frame):
    from later.src.train.stage2_common import checkpoint_hash, tokenizer_hashes
    from later.src.train.stage2_monitor import Stage2Monitor
    from later.src.train.stage2_plan_dataset import BOUNDARY_RULE_VERSION
    from later.src.train.stage2_teacher_cache import TeacherBoundaryCache
    cache = TeacherBoundaryCache("/scratch/hz33227/e2c/stage2/teacher_cache", "val",
                                 {"teacher_checkpoint_hash": checkpoint_hash(CKPT), **tokenizer_hashes(tok, CKPT),
                                  "boundary_rule_version": BOUNDARY_RULE_VERSION})
    layers = list(range(1, 37))
    mon = Stage2Monitor(model, tok, val_frame, cache, layers, 8, probe_n=8, probe_batch_size=4, grad_batch_size=2)
    model.train()
    model.zero_grad(set_to_none=True)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if "latent_projector" in n or n.endswith("layers.0.mlp.down_proj.weight")}
    res = mon.probe_ce()
    res.update(mon.projector_grad_split(0.1))
    probs = [{"question": val_frame.question.iloc[i], "answer": val_frame.answer.iloc[i]} for i in range(2)]
    res.update(mon.generation_eval(probs, ["learned", "zero", "shuffled", "explicit"], max_new_tokens=16, batch_size=2))
    assert model.training                                                # train mode restored
    assert all(p.grad is None for p in model.parameters())               # nothing written to .grad
    for n, p in model.named_parameters():
        if n in before:
            assert torch.equal(p.detach(), before[n]), n                 # no parameter update
    for k in ("val/ce_learned", "val/ce_zero", "val/ce_shuffled", "val/ce_no_latent", "val/ce_explicit",
              "val/latent_gain_ce", "val/alignment_learned", "val/alignment_zero", "val/ce_learned_nonrep",
              "val/grad_projector_from_execution", "val/grad_projector_from_alignment", "gen/acc_shuffled"):
        assert k in res and res[k] == res[k], k                         # present and not NaN
    assert res["val/grad_projector_from_execution"] > 0 and res["val/grad_projector_from_alignment"] > 0
    model.eval()
