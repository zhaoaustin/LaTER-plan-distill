"""Entry point for OUR Stage 2: objective_mode=plan_distill_stage2 (deterministic latent plan distillation).

    torchrun --nproc-per-node=N -m later.src.train.train_stage2 --config later/src/config/stage2_plan_kd_qwen3.yaml

This is NOT LaTER's internal train_stage=2 curriculum. It reuses LaTER's LatentQwenForCausalLM, LatentProjector
and recursive latent builder (made differentiable), and adds the three-term Stage-2 objective
(see later/src/train/stage2_objective.py). The LaTER baseline entry points (train.py/trainer.py) are unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Any, Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from later.src.train.stage2_common import checkpoint_hash, dump_json, load_stage2_model, load_tokenizer, tokenizer_hashes
from later.src.train.stage2_objective import Stage2Module
from later.src.train.stage2_plan_dataset import BOUNDARY_RULE_VERSION, Stage2Collator, Stage2PlanDataset
from later.src.train.stage2_teacher_cache import TeacherBoundaryCache
from later.src.train.utils import load_yaml

REQUIRED = ["objective_mode", "model_path", "reference_teacher_path", "train_data", "val_data", "reference_cache_dir",
            "output_dir", "data_source_policy", "allow_later69k_mixing", "require_stage1_data_audit", "latent_steps",
            "latent_embed_builder_no_grad", "alignment_layers", "alignment_distance", "alignment_scale",
            "alignment_epsilon", "lambda_a", "lambda_r", "execution_ce_weight", "kl_loss_weight",
            "teacher_kl_enabled", "halting_loss_enabled", "noise_std", "rl_enabled"]


def validate_config(cfg: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"config missing keys: {missing}")
    assert cfg["objective_mode"] == "plan_distill_stage2", cfg["objective_mode"]
    assert cfg["data_source_policy"] == "stage1_sft_only"
    assert cfg["allow_later69k_mixing"] is False
    assert cfg["latent_embed_builder_no_grad"] is False, "Stage 2 requires a differentiable latent builder"
    assert cfg["alignment_distance"] == "normalized_l1" and cfg["alignment_scale"] == "teacher_std"
    assert float(cfg["noise_std"]) == 0.0 and not cfg["rl_enabled"]
    assert cfg["model_path"] == cfg["reference_teacher_path"] or cfg.get("resume_from"), \
        "student must be initialised from the same Stage-1 checkpoint as the reference teacher"
    if cfg["require_stage1_data_audit"]:
        audit_path = os.path.join(os.path.dirname(cfg["train_data"]), "stage1_to_stage2_data_audit.json")
        audit = json.load(open(audit_path))
        assert audit["later69k_samples_used"] == 0 and audit["data_source_policy"] == "stage1_sft_only"
        assert audit["n_accepted"]["train"] > 0 and audit["checkpoint_path"] == cfg["reference_teacher_path"], audit_path


def setup_dist():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    return rank, world, local


def all_mean(x: float, device) -> float:
    if not dist.is_initialized():
        return float(x)
    t = torch.tensor(float(x), device=device)
    dist.all_reduce(t)
    return float(t) / dist.get_world_size()


def grad_norm(params) -> float:
    gs = [p.grad.detach().float() for p in params if p.grad is not None]
    return float(torch.sqrt(sum((g ** 2).sum() for g in gs))) if gs else 0.0


def cosine_lr(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


@torch.no_grad()
def evaluate(module, loader, cache, layer_ids, device, max_batches):
    module.eval()
    sums, n = {"l_exec": 0.0, "l_align": 0.0, "l_explicit": 0.0, "loss": 0.0}, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        th = cache.get(batch["record_ids"], layer_ids, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = module(batch, th)
        for k in sums:
            sums[k] += float(out[k])
        n += 1
    module.train()
    return {f"val/{k}": all_mean(v / max(n, 1), device) for k, v in sums.items()}


def save(cfg, model, tok, optimizer, step, rank, layer_ids, tag=None):
    if rank != 0:
        return
    d = os.path.join(cfg["output_dir"], tag or f"checkpoint-{step}")
    os.makedirs(d, exist_ok=True)
    model.save_pretrained(d, safe_serialization=True)
    tok.save_pretrained(d)
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state()}, os.path.join(d, "trainer_state.pt"))
    dump_json({"objective_mode": "plan_distill_stage2", "step": step, "latent_steps": cfg["latent_steps"],
               "alignment_layers": layer_ids, "lambda_a": cfg["lambda_a"], "lambda_r": cfg["lambda_r"],
               "boundary_rule_version": BOUNDARY_RULE_VERSION, "init_checkpoint": cfg["model_path"]},
              os.path.join(d, "stage2_meta.json"))
    with open(os.path.join(cfg["output_dir"], "latest_checkpoint.txt"), "w") as f:
        f.write(os.path.basename(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="key=value (yaml-parsed)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    import yaml
    for kv in args.override:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    validate_config(cfg)
    rank, world, local = setup_dist()
    device = torch.device("cuda", local)
    seed = int(cfg.get("seed", 1234))
    random.seed(seed); torch.manual_seed(seed)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    resume = None
    init_path = cfg["model_path"]
    latest = os.path.join(cfg["output_dir"], "latest_checkpoint.txt")
    if cfg.get("resume", True) and os.path.exists(latest):
        resume = os.path.join(cfg["output_dir"], open(latest).read().strip())
        init_path = resume
    tok = load_tokenizer(cfg["model_path"])
    model = load_stage2_model(init_path, latent_embed_builder_no_grad=cfg["latent_embed_builder_no_grad"],
                              latent_projector_dropout=float(cfg.get("latent_projector_dropout", 0.0)),
                              dtype=getattr(torch, cfg.get("master_dtype", "float32")),
                              attn_implementation=cfg.get("attn_implementation", "sdpa"),
                              latent_output_rms=cfg.get("latent_projector_output_rms")).to(device)
    assert model.config.latent_embed_builder_no_grad is False
    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    layer_ids = model.resolve_alignment_layers(cfg["alignment_layers"], model.config.num_hidden_layers)

    expect = {"teacher_checkpoint_hash": checkpoint_hash(cfg["reference_teacher_path"]),
              **tokenizer_hashes(tok, cfg["reference_teacher_path"]), "boundary_rule_version": BOUNDARY_RULE_VERSION}
    train_cache = TeacherBoundaryCache(cfg["reference_cache_dir"], "train", expect)
    val_cache = TeacherBoundaryCache(cfg["reference_cache_dir"], "val", expect)
    assert set(layer_ids) <= set(train_cache.layer_ids)

    L = int(cfg["latent_steps"])
    train_ds = Stage2PlanDataset(cfg["train_data"], tok, L, int(cfg.get("max_length", 4096)),
                                 limit=cfg.get("train_limit") or None)
    val_ds = Stage2PlanDataset(cfg["val_data"], tok, L, int(cfg.get("max_length", 4096)),
                               limit=cfg.get("val_limit") or None)
    collate = Stage2Collator(tok.pad_token_id)
    mbs = int(cfg.get("micro_batch_size", 1))
    accum = int(cfg.get("gradient_accumulation_steps", 1))
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=seed, drop_last=True)
    train_loader = DataLoader(train_ds, batch_size=mbs, sampler=sampler, collate_fn=collate, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=mbs, shuffle=False, collate_fn=collate, num_workers=2,
                            sampler=DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False))

    module = Stage2Module(model, cfg, layer_ids)
    ddp = DDP(module, device_ids=[local]) if world > 1 else module
    proj_params = list(model.latent_projector.parameters())
    proj_ids = {id(p) for p in proj_params}
    back_params = [p for p in model.parameters() if id(p) not in proj_ids]
    optimizer = torch.optim.AdamW(
        [{"params": back_params, "lr": float(cfg["learning_rate"])},
         {"params": proj_params, "lr": float(cfg.get("latent_projector_lr", cfg["learning_rate"]))}],
        betas=tuple(cfg.get("adam_betas", [0.9, 0.95])), weight_decay=float(cfg.get("weight_decay", 0.01)))
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    max_steps = int(cfg["max_train_steps"])
    warmup = int(cfg.get("warmup_steps", 0))
    step = 0
    if resume:
        st = torch.load(os.path.join(resume, "trainer_state.pt"), map_location="cpu", weights_only=False)
        optimizer.load_state_dict(st["optimizer"])
        step = int(st["step"])
    log_f = open(os.path.join(cfg["output_dir"], f"train_log_rank{rank}.jsonl"), "a") if rank == 0 else None
    if rank == 0:
        dump_json({"config": cfg, "layer_ids": layer_ids,
                   "latent_projector_output_rms_value": model.latent_projector.output_rms, "world_size": world, "global_batch": mbs * accum * world,
                   "n_train": len(train_ds), "n_val": len(val_ds), "resume_from": resume,
                   "latent_embed_builder_no_grad": model.config.latent_embed_builder_no_grad,
                   "teacher_cache_manifest": {k: train_cache.manifest[k] for k in train_cache.manifest if k != "splits"},
                   "torch": torch.__version__, "transformers": __import__("transformers").__version__},
                  os.path.join(cfg["output_dir"], "run_config_resolved.json"))
        print(json.dumps({"start_step": step, "max_steps": max_steps, "global_batch": mbs * accum * world}), flush=True)
    debug_every = int(cfg.get("debug_latent_grad_every", 10))
    epoch = 0
    it = None
    t_last = time.time()
    while step < max_steps:
        if it is None:
            sampler.set_epoch(epoch)
            it = iter(train_loader)
            # resume: skip already-consumed micro-batches of this epoch
            per_epoch = len(train_loader) // accum
            skip = (step % max(per_epoch, 1)) * accum if resume and epoch == 0 else 0
            for _ in range(skip):
                next(it)
        torch.cuda.reset_peak_memory_stats()
        f = cosine_lr(step, max_steps, warmup)
        for g, lr0 in zip(optimizer.param_groups, base_lrs):
            g["lr"] = lr0 * f
        agg = {"l_exec": 0.0, "l_align": 0.0, "l_explicit": 0.0, "loss": 0.0, "tE": 0, "tA": 0, "tP": 0, "tC": 0,
               "scale": 0.0, "l1": 0.0, "s_len": 0, "t_len": 0, "early": 0.0}
        debug = debug_every > 0 and step % debug_every == 0
        for micro in range(accum):
            try:
                batch = next(it)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                it = iter(train_loader)
                batch = next(it)
            th = train_cache.get(batch["record_ids"], layer_ids, device)
            sync = (micro == accum - 1) or world == 1
            ctx = ddp.no_sync() if (world > 1 and not sync) else torch.enable_grad()
            with ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = ddp(batch, th, debug_latent_grad=debug and micro == 0)
                (out["loss"] / accum).backward()
            for k in ("l_exec", "l_align", "l_explicit", "loss"):
                agg[k] += float(out[k]) / accum
            agg["tE"] += int((batch["student_seg_E"] & (batch["student_labels"] != -100)).sum())
            agg["tA"] += int((batch["student_seg_A"] & (batch["student_labels"] != -100)).sum())
            agg["tC"] += int((batch["student_seg_C"] & (batch["student_labels"] != -100)).sum())
            agg["tP"] += int((batch["explicit_seg_P"] & (batch["explicit_labels"] != -100)).sum())
            agg["s_len"] += int(batch["student_boundary"].max()) + 1
            agg["t_len"] += int(batch["teacher_boundary"].max()) + 1
            if "align_info" in out:
                agg["scale"] += float(out["align_info"]["teacher_scale_mean"]) / accum
                agg["l1"] += float(out["align_info"]["boundary_l1_mean"]) / accum
            if debug and micro == 0 and out.get("latent", {}).get("projector_outputs"):
                z1 = out["latent"]["projector_outputs"][0]
                agg["early"] = float(z1.grad.float().norm()) if z1.grad is not None else float("nan")
        gn_back = grad_norm(back_params)
        gn_proj = grad_norm(proj_params)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("max_grad_norm", 1.0)))
        if not torch.isfinite(total_norm):
            raise FloatingPointError(f"non-finite grad norm at step {step}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        rec = {"step": step, "lr": optimizer.param_groups[0]["lr"], "lr_projector": optimizer.param_groups[1]["lr"],
               "batch_size": mbs * accum * world, "supervised_E_tokens": agg["tE"], "supervised_A_tokens": agg["tA"],
               "supervised_close_tokens": agg["tC"], "supervised_P_tokens": agg["tP"],
               "L_execution": all_mean(agg["l_exec"], device), "L_alignment": all_mean(agg["l_align"], device),
               "L_explicit": all_mean(agg["l_explicit"], device), "lambda_a": float(cfg["lambda_a"]),
               "lambda_r": float(cfg["lambda_r"]), "L_total": all_mean(agg["loss"], device),
               "grad_norm_backbone": gn_back, "grad_norm_latent_projector": gn_proj,
               "grad_norm_total_preclip": float(total_norm), "early_latent_grad_norm": agg["early"] if debug else None,
               "teacher_boundary_scale_mean": all_mean(agg["scale"], device),
               "boundary_hidden_l1_mean": all_mean(agg["l1"], device), "latent_steps": L,
               "student_prompt_length": agg["s_len"] / accum, "teacher_prefix_length": agg["t_len"] / accum,
               "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2 ** 30,
               "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2 ** 30,
               "explicit_branch_CE": all_mean(agg["l_explicit"], device), "latent_branch_CE": all_mean(agg["l_exec"], device),
               "step_time_s": time.time() - t_last}
        t_last = time.time()
        if int(cfg.get("eval_every_steps", 0)) and step % int(cfg["eval_every_steps"]) == 0:
            rec.update(evaluate(module, val_loader, val_cache, layer_ids, device, int(cfg.get("eval_max_batches", 50))))
        if rank == 0:
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            if step % int(cfg.get("log_every_steps", 1)) == 0:
                print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        if step % int(cfg.get("save_every_steps", 100)) == 0 or step == max_steps:
            if dist.is_initialized():
                dist.barrier()
            save(cfg, model, tok, optimizer, step, rank, layer_ids)
            if dist.is_initialized():
                dist.barrier()
    if rank == 0:
        print("TRAINING_DONE", step, flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
