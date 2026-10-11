"""Entry point for the two-loss PlanRec Stage 2 pilot (objective_mode=planrec_two_loss).

    torchrun --nproc-per-node=2 -m later.src.train.train_stage2_planrec --config later/src/config/stage2_planrec_two_loss.yaml

L_total = execution_ce_weight * L_execution + lambda_p * L_planrec (see stage2_planrec_objective.py). Starts fresh
from the Stage-1 SFT checkpoint with a seed-initialised latent projector; it refuses to start from a three-loss
Stage-2 checkpoint. Generic pieces (DDP setup, cosine LR, all-reduce, git state) are reused from train_stage2.py,
which is unchanged. No teacher boundary-hidden or KL cache is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from typing import Any, Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from later.src.train.stage2_common import checkpoint_hash, dump_json, file_sha256, load_stage2_model, load_tokenizer, tokenizer_hashes
from later.src.train.stage2_plan_dataset import (BOUNDARY_RULE_VERSION, PLAN_CONTENT_RULE, PLANREC_RULE_VERSION,
                                                 Stage2Collator, Stage2PlanDataset)
from later.src.train.stage2_planrec_monitor import PlanRecMonitor
from later.src.train.stage2_planrec_objective import PlanRecModule, check_disabled
from later.src.train.train_stage2 import all_mean, cosine_lr, git_state, grad_norm, setup_dist
from later.src.train.utils import load_yaml

REQUIRED = ["objective_mode", "model_path", "stage1_hf_repo", "stage1_hf_revision", "train_data", "val_data",
            "output_dir", "latent_steps", "latent_embed_builder_no_grad", "latent_projector_output_rms",
            "execution_ce_weight", "lambda_p", "seed", "max_train_steps", "data_source_policy", "allow_later69k_mixing"]


def validate_config(cfg: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"config missing keys: {missing}")
    assert cfg["objective_mode"] == "planrec_two_loss", cfg["objective_mode"]
    assert cfg["data_source_policy"] == "stage1_sft_only" and cfg["allow_later69k_mixing"] is False
    assert cfg["latent_embed_builder_no_grad"] is False, "PlanRec needs a differentiable latent builder"
    assert cfg["latent_projector_output_rms"] == "token_embedding"
    assert float(cfg["lambda_p"]) >= 0.0 and float(cfg["execution_ce_weight"]) == 1.0
    check_disabled(cfg)
    assert not cfg.get("resume_from_checkpoint"), "PlanRec pilot must start fresh from Stage 1 (no old Stage-2 ckpt)"
    audit = json.load(open(os.path.join(os.path.dirname(cfg["train_data"]), "stage1_to_stage2_data_audit.json")))
    assert audit["later69k_samples_used"] == 0 and audit["data_source_policy"] == "stage1_sft_only"
    assert audit["checkpoint_path"] == cfg["model_path"], "data audit was made against a different Stage-1 checkpoint"


def param_hash(params) -> str:
    h = hashlib.sha256()
    for p in params:
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def save(cfg, model, tok, optimizer, step, rank, sampler_state, weights_only=False):
    if rank != 0:
        return
    d = os.path.join(cfg["output_dir"], f"checkpoint-{step}")
    os.makedirs(d, exist_ok=True)
    model.save_pretrained(d, safe_serialization=True)
    tok.save_pretrained(d)
    if not weights_only:
        torch.save({"step": step, "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(), "python_rng": random.getstate(),
                    "sampler_state": sampler_state,
                    "lr_schedule": {"type": "cosine_with_linear_warmup", "max_steps": int(cfg.get("lr_schedule_max_steps", cfg["max_train_steps"])),
                                    "warmup_steps": int(cfg.get("warmup_steps", 0)),
                                    "base_lrs": [float(cfg["learning_rate"]), float(cfg["latent_projector_lr"])],
                                    "lr_at_save": [g["lr"] for g in optimizer.param_groups]}},
                   os.path.join(d, "trainer_state.pt"))
    dump_json({"objective_mode": "planrec_two_loss", "step": step, "latent_steps": cfg["latent_steps"],
               "lambda_p": cfg["lambda_p"], "execution_ce_weight": cfg["execution_ce_weight"],
               "weights_only": weights_only, "boundary_rule_version": BOUNDARY_RULE_VERSION,
               "planrec_rule_version": PLANREC_RULE_VERSION, "init_checkpoint": cfg["model_path"],
               "stage1_hf_revision": cfg["stage1_hf_revision"]}, os.path.join(d, "stage2_meta.json"))
    if not weights_only:
        with open(os.path.join(cfg["output_dir"], "latest_checkpoint.txt"), "w") as f:
            f.write(os.path.basename(d))


def init_wandb(cfg, model, train_ds, val_ds, world, mbs, accum, monitor, start_info):
    import transformers
    import wandb
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    wcfg = {"objective_mode": cfg["objective_mode"], "run_group": cfg.get("run_label"),
            "loss": {"formula": "execution_ce_weight*L_execution + lambda_p*L_planrec",
                     "execution_ce_weight": float(cfg["execution_ce_weight"]), "lambda_p": float(cfg["lambda_p"]),
                     "lambda_a": 0.0, "lambda_r": 0.0, "execution_kl": False, "teacher_kl": False, "halting": False,
                     "noise_std": 0.0, "rl": False, "planrec_rule": PLANREC_RULE_VERSION, "content_rule": PLAN_CONTENT_RULE},
            "stage1_checkpoint": {"path": cfg["model_path"], "hf_repo": cfg["stage1_hf_repo"],
                                  "hf_revision": cfg["stage1_hf_revision"], "checkpoint_hash": checkpoint_hash(cfg["model_path"])},
            "git": git_state(repo_root),
            "dataset": {"train_parquet": cfg["train_data"], "train_sha256": file_sha256(cfg["train_data"]),
                        "val_parquet": cfg["val_data"], "val_sha256": file_sha256(cfg["val_data"]),
                        "n_train": len(train_ds), "n_val": len(val_ds), "later69k_samples": 0},
            "latent_steps": int(cfg["latent_steps"]), "projector_output_rms": model.latent_projector.output_rms,
            "learning_rate": float(cfg["learning_rate"]), "latent_projector_lr": float(cfg["latent_projector_lr"]),
            "global_batch": mbs * accum * world, "micro_batch_size": mbs, "gradient_accumulation_steps": accum,
            "world_size": world, "seed": int(cfg["seed"]), "max_train_steps": int(cfg["max_train_steps"]),
            "probe_val_record_ids": monitor.probe_ids, "start": start_info,
            "versions": {"transformers": transformers.__version__, "torch": torch.__version__}}
    id_file = os.path.join(cfg["output_dir"], "wandb_run_id.txt")
    run_id = open(id_file).read().strip() if os.path.exists(id_file) else __import__("uuid").uuid4().hex[:12]
    open(id_file, "w").write(run_id)
    return wandb.init(project=cfg.get("wandb_project", "stage2-plan-reconstruction"), entity=cfg.get("wandb_entity"),
                      group=cfg.get("wandb_group"), tags=cfg.get("wandb_tags"),
                      name=cfg.get("wandb_run_name", os.path.basename(cfg["output_dir"])), id=run_id, resume="allow",
                      config=wcfg, dir=cfg["output_dir"])


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
    seed = int(cfg["seed"])
    random.seed(seed); torch.manual_seed(seed)          # => identical projector init for every run with this seed
    os.makedirs(cfg["output_dir"], exist_ok=True)

    resume, init_path = None, cfg["model_path"]
    latest = os.path.join(cfg["output_dir"], "latest_checkpoint.txt")
    if cfg.get("resume", False) and os.path.exists(latest):   # only this run's own checkpoints
        resume = os.path.join(cfg["output_dir"], open(latest).read().strip())
        meta = json.load(open(os.path.join(resume, "stage2_meta.json")))
        assert meta["objective_mode"] == "planrec_two_loss" and float(meta["lambda_p"]) == float(cfg["lambda_p"]), meta
        init_path = resume
    tok = load_tokenizer(cfg["model_path"])
    model = load_stage2_model(init_path, latent_embed_builder_no_grad=False,
                              latent_projector_dropout=float(cfg.get("latent_projector_dropout", 0.0)),
                              dtype=getattr(torch, cfg.get("master_dtype", "float32")),
                              attn_implementation=cfg.get("attn_implementation", "sdpa"),
                              latent_output_rms=cfg["latent_projector_output_rms"]).to(device)
    assert model.config.latent_embed_builder_no_grad is False
    model.train()
    L = int(cfg["latent_steps"])
    train_ds = Stage2PlanDataset(cfg["train_data"], tok, L, int(cfg.get("max_length", 4096)), limit=cfg.get("train_limit") or None)
    val_ds = Stage2PlanDataset(cfg["val_data"], tok, L, int(cfg.get("max_length", 4096)))
    collate = Stage2Collator(tok.pad_token_id)
    mbs, accum = int(cfg.get("micro_batch_size", 1)), int(cfg.get("gradient_accumulation_steps", 1))
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=seed, drop_last=True)
    train_loader = DataLoader(train_ds, batch_size=mbs, sampler=sampler, collate_fn=collate, num_workers=4)

    module = PlanRecModule(model, cfg)
    ddp = DDP(module, device_ids=[local]) if world > 1 else module
    proj_params = list(model.latent_projector.parameters())
    proj_ids = {id(p) for p in proj_params}
    back_params = [p for p in model.parameters() if id(p) not in proj_ids]
    optimizer = torch.optim.AdamW(
        [{"params": back_params, "lr": float(cfg["learning_rate"])},
         {"params": proj_params, "lr": float(cfg["latent_projector_lr"])}],
        betas=tuple(cfg.get("adam_betas", [0.9, 0.95])), weight_decay=float(cfg.get("weight_decay", 0.01)))
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    max_steps = int(cfg["max_train_steps"])
    sched_steps = int(cfg.get("lr_schedule_max_steps", max_steps))
    warmup = int(cfg.get("warmup_steps", 0))
    step = 0
    if resume:
        st = torch.load(os.path.join(resume, "trainer_state.pt"), map_location="cpu", weights_only=False)
        optimizer.load_state_dict(st["optimizer"])
        step = int(st["step"])
        torch.set_rng_state(st["rng"])
        random.setstate(st["python_rng"])
    start_info = {"start_step": step, "resumed_from": resume, "seed": seed}
    if rank == 0:
        start_info.update({"initial_projector_param_hash": param_hash(proj_params),
                           "initial_model_param_hash": param_hash(list(model.parameters())),
                           "stage1_checkpoint_hash": checkpoint_hash(cfg["model_path"]),
                           **tokenizer_hashes(tok, cfg["model_path"])})
        dump_json(start_info, os.path.join(cfg["output_dir"], "start_provenance.json" if not resume else f"resume_provenance_{step}.json"))
        dump_json({"config": cfg, "world_size": world, "global_batch": mbs * accum * world, "n_train": len(train_ds),
                   "n_val": len(val_ds), "projector_output_rms": model.latent_projector.output_rms,
                   "planrec_rule_version": PLANREC_RULE_VERSION, "plan_content_rule": PLAN_CONTENT_RULE,
                   "torch": torch.__version__, "transformers": __import__("transformers").__version__},
                  os.path.join(cfg["output_dir"], "run_config_resolved.json"))
        if not resume:   # step-0 projector (the backbone at step 0 is the Stage-1 checkpoint itself)
            torch.save(model.latent_projector.state_dict(), os.path.join(cfg["output_dir"], "projector_step0.pt"))
        print(json.dumps(start_info), flush=True)
    log_f = open(os.path.join(cfg["output_dir"], "train_log_rank0.jsonl"), "a") if rank == 0 else None
    batch_log = open(os.path.join(cfg["output_dir"], f"batch_record_ids_rank{rank}.jsonl"), "a")
    probe_frame = val_ds.frame if not cfg.get("probe_data") else __import__("pandas").read_parquet(cfg["probe_data"])
    monitor = PlanRecMonitor(model, tok, probe_frame, L, probe_n=int(cfg.get("probe_n", 128)),
                             probe_batch_size=int(cfg.get("probe_batch_size", 8)),
                             grad_batch_size=int(cfg.get("grad_split_batch_size", 4)), lambda_p=float(cfg["lambda_p"]))
    wb = init_wandb(cfg, model, train_ds, val_ds, world, mbs, accum, monitor, start_info) \
        if rank == 0 and cfg.get("wandb", False) else None

    def wlog(d, at):
        if wb is not None:
            wb.log(d, step=at)
        if rank == 0:
            log_f.write(json.dumps({"step": at, **d}) + "\n"); log_f.flush()

    probe_steps = set(int(x) for x in (cfg.get("probe_steps") or []))

    def run_probe(at):
        t0 = time.time()
        res = monitor.probe()
        if rank == 0:
            res.update(monitor.grad_split())
            res["val/probe_seconds"] = time.time() - t0
        if dist.is_initialized():
            dist.barrier()
        wlog(res, at)
        if rank == 0:
            print(json.dumps({k: round(v, 5) for k, v in res.items() if isinstance(v, float)}), flush=True)

    if step in probe_steps:
        run_probe(step)
    save_steps = set(int(x) for x in (cfg.get("save_steps") or []))
    weights_only_steps = set(int(x) for x in (cfg.get("save_weights_only_steps") or []))
    loss_every, grad_every = int(cfg.get("wandb_loss_every", 10)), int(cfg.get("wandb_grad_every", 20))
    win, win_n = {"loss_total": 0.0, "loss_execution": 0.0, "loss_planrec": 0.0}, 0
    epoch = step * accum // max(len(train_loader), 1)
    it = None
    lambda_p = float(cfg["lambda_p"])
    while step < max_steps:
        if it is None:
            sampler.set_epoch(epoch)
            it = iter(train_loader)
            skip = (step * accum) % max(len(train_loader), 1) if resume else 0
            for _ in range(skip):
                next(it)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        f = cosine_lr(step, sched_steps, warmup)
        for g, lr0 in zip(optimizer.param_groups, base_lrs):
            g["lr"] = lr0 * f
        agg = {"l_exec": 0.0, "l_rec": 0.0, "loss": 0.0, "n_exec": 0, "n_rec": 0}
        z1_info: Dict[str, float] = {}
        step_rids = []
        debug = grad_every > 0 and (step + 1) % grad_every == 0
        for micro in range(accum):
            try:
                batch = next(it)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                it = iter(train_loader)
                batch = next(it)
            step_rids += list(batch["record_ids"])
            sync = (micro == accum - 1) or world == 1
            ctx = ddp.no_sync() if (world > 1 and not sync) else torch.enable_grad()
            with ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = ddp(batch, debug_latent_grad=debug and micro == 0)
                if debug and micro == 0:
                    # gradient reaching the earliest latent z_1 from each loss separately (autograd.grad on a
                    # non-leaf input: no parameter .grad written, so DDP's reducer is not involved)
                    z1 = out["latent"]["projector_outputs"][0]
                    g_e = torch.autograd.grad(out["l_exec"], z1, retain_graph=True)[0]
                    z1_info["train/grad_z1_from_exec"] = float(g_e.float().norm())
                    if lambda_p > 0:
                        g_r = torch.autograd.grad(lambda_p * out["l_planrec"], z1, retain_graph=True)[0]
                        z1_info["train/grad_z1_from_planrec"] = float(g_r.float().norm())
                        z1_info["train/cos_z1_planrec_vs_exec"] = float(torch.nn.functional.cosine_similarity(
                            g_r.float().flatten(), g_e.float().flatten(), dim=0))
                (out["loss"] / accum).backward()
            agg["l_exec"] += float(out["l_exec"]) / accum
            agg["l_rec"] += float(out["l_planrec"]) / accum
            agg["loss"] += float(out["loss"]) / accum
            agg["n_exec"] += int(out["n_exec_tokens"])
            agg["n_rec"] += int(out.get("n_planrec_tokens", 0))
        gn_back, gn_proj = grad_norm(back_params), grad_norm(proj_params)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("max_grad_norm", 1.0)))
        if not torch.isfinite(total_norm):
            raise FloatingPointError(f"non-finite grad norm at step {step}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        step_time = time.time() - t0
        batch_log.write(json.dumps({"step": step, "record_ids": step_rids}) + "\n"); batch_log.flush()
        bs = out["latent"]["builder_stats"]
        rec = {"step": step, "lr": optimizer.param_groups[0]["lr"], "lr_projector": optimizer.param_groups[1]["lr"],
               "L_execution": all_mean(agg["l_exec"], device), "L_planrec": all_mean(agg["l_rec"], device),
               "L_total": all_mean(agg["loss"], device), "lambda_p": lambda_p,
               "exec_tokens_rank": agg["n_exec"], "planrec_tokens_rank": agg["n_rec"],
               "grad_norm_backbone": gn_back, "grad_norm_latent_projector": gn_proj, "grad_norm_total_preclip": float(total_norm),
               "latent_rms_ratio": bs.get("latent_rms", float("nan")) / max(bs.get("token_embedding_rms", float("nan")), 1e-12),
               "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2 ** 30, "step_time_s": step_time, **z1_info}
        if rank == 0:
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        for k, v in (("loss_total", rec["L_total"]), ("loss_execution", rec["L_execution"]), ("loss_planrec", rec["L_planrec"])):
            win[k] += v
        win_n += 1
        if step % loss_every == 0:
            if wb is not None:
                m = {f"train/{k}": v / win_n for k, v in win.items()}
                m.update({"train/loss_planrec_weighted": lambda_p * m["train/loss_planrec"], "train/learning_rate": rec["lr"],
                          "train/learning_rate_projector": rec["lr_projector"]})
                wb.log(m, step=step)
            win, win_n = {k: 0.0 for k in win}, 0
        if wb is not None and step % grad_every == 0:
            wb.log({"train/grad_projector": gn_proj, "train/grad_backbone": gn_back, "train/grad_total_preclip": float(total_norm),
                    "train/latent_rms_ratio": rec["latent_rms_ratio"], "train/step_time": step_time,
                    "train/peak_gpu_memory": rec["peak_allocated_GiB"], **z1_info}, step=step)
        if step in probe_steps:
            run_probe(step)
        if step in save_steps or step == max_steps or step in weights_only_steps:
            if dist.is_initialized():
                dist.barrier()
            save(cfg, model, tok, optimizer, step, rank,
                 sampler_state={"sampler": "DistributedSampler", "seed": seed, "epoch": epoch, "world_size": world,
                                "micro_batches_consumed_per_rank": step * accum, "micro_batch_size": mbs,
                                "gradient_accumulation_steps": accum, "drop_last": True},
                 weights_only=step in weights_only_steps and step not in save_steps and step != max_steps)
            if dist.is_initialized():
                dist.barrier()
    # DDP consistency check: every rank must hold identical parameters after the last optimizer step
    chk = {"rank": rank, "step": step, "projector_hash": param_hash(proj_params),
           "layer17_down_proj_hash": param_hash([model.model.layers[17].mlp.down_proj.weight])}
    print("RANK_PARAM_HASH", json.dumps(chk), flush=True)
    if rank == 0:
        print("TRAINING_DONE", step, flush=True)
        if wb is not None:
            wb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
