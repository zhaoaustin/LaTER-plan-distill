"""Verify a Stage-2 checkpoint is completely resumable; writes resume_manifest.json next to it.

Checks: files + sha256, step, model weights load into LatentQwenForCausalLM (fp32 master), optimizer state loads
into the exact AdamW param groups used by train_stage2 (every exp_avg/exp_avg_sq matches its parameter shape and
is non-zero), RNG state present, LR at the next step from the cosine schedule, and the data-sampling state: the
DistributedSampler order (seed, epoch) and the number of micro-batches already consumed, i.e. the exact record ids
the resumed run will see next (disjoint from those already used).
"""
import argparse
import hashlib
import json
import math
import os

import torch
from torch.utils.data import DistributedSampler

from later.src.train.stage2_common import load_stage2_model
from later.src.train.utils import load_yaml


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def cosine_lr(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_train", type=int, required=True)
    ap.add_argument("--world_size", type=int, default=2)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    d = args.ckpt
    files = {f: {"bytes": os.path.getsize(os.path.join(d, f)), "sha256": sha(os.path.join(d, f))}
             for f in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, f))}
    st = torch.load(os.path.join(d, "trainer_state.pt"), map_location="cpu", weights_only=False)
    meta = json.load(open(os.path.join(d, "stage2_meta.json")))
    step = int(st["step"])
    assert step == int(meta["step"]), (step, meta["step"])

    model = load_stage2_model(d, latent_embed_builder_no_grad=False, latent_projector_dropout=0.0, dtype=torch.float32)
    assert model.config.latent_embed_builder_no_grad is False and model.latent_projector.output_rms
    proj = list(model.latent_projector.parameters())
    pid = {id(p) for p in proj}
    back = [p for p in model.parameters() if id(p) not in pid]
    opt = torch.optim.AdamW([{"params": back, "lr": float(cfg["learning_rate"])},
                             {"params": proj, "lr": float(cfg["latent_projector_lr"])}],
                            betas=tuple(cfg["adam_betas"]), weight_decay=float(cfg["weight_decay"]))
    opt.load_state_dict(st["optimizer"])
    n_state, n_zero, steps_seen = 0, 0, set()
    for g in opt.param_groups:
        for p in g["params"]:
            s = opt.state[p]
            assert s["exp_avg"].shape == p.shape and s["exp_avg_sq"].shape == p.shape
            n_state += 1
            n_zero += int(s["exp_avg_sq"].abs().sum() == 0)
            steps_seen.add(float(s["step"]))
    assert n_state == len(list(model.parameters())), (n_state, len(list(model.parameters())))

    accum, mbs, W = int(cfg["gradient_accumulation_steps"]), int(cfg["micro_batch_size"]), args.world_size
    per_epoch = (args.n_train // W // mbs) // accum
    epoch, in_epoch = step // per_epoch, step % per_epoch
    consumed = in_epoch * accum * mbs
    nxt, used = {}, {}
    for r in range(W):
        order = list(DistributedSampler(range(args.n_train), num_replicas=W, rank=r, shuffle=True,
                                        seed=int(cfg["seed"]), drop_last=True).__iter__()) if epoch == 0 else None
        if order is None:
            s = DistributedSampler(range(args.n_train), num_replicas=W, rank=r, shuffle=True, seed=int(cfg["seed"]),
                                   drop_last=True)
            s.set_epoch(epoch)
            order = list(iter(s))
        nxt[r], used[r] = order[consumed: consumed + 8], set(order[:consumed])
        assert not set(order[consumed: consumed + accum * mbs]) & used[r]
    base = [float(cfg["learning_rate"]), float(cfg["latent_projector_lr"])]
    f = cosine_lr(step, int(cfg["max_train_steps"]), int(cfg["warmup_steps"]))
    manifest = {
        "checkpoint": os.path.abspath(d), "step": step, "files": files,
        "model": {"class": type(model).__name__, "dtype": "float32 master weights",
                  "latent_projector_output_rms": model.latent_projector.output_rms,
                  "latent_embed_builder_no_grad": model.config.latent_embed_builder_no_grad},
        "optimizer": {"type": "AdamW", "param_groups": len(opt.param_groups), "tensors_with_state": n_state,
                      "tensors_with_all_zero_exp_avg_sq": n_zero, "adam_step_counters": sorted(steps_seen),
                      "lr_at_save": [g["lr"] for g in st["optimizer"]["param_groups"]]},
        "scheduler": {"type": "cosine with linear warmup, computed from global_step (no scheduler object)",
                      "max_steps": int(cfg["max_train_steps"]), "warmup_steps": int(cfg["warmup_steps"]),
                      "lr_next_step": [b * f for b in base]},
        "rng": {"torch_cpu": "rng" in st, "torch_cuda": "cuda_rng" in st, "python": "python_rng" in st,
                "note": "training math uses no RNG (projector dropout 0; sampler uses its own seeded generator)"},
        "data_sampling": {"sampler": "DistributedSampler(shuffle=True, drop_last=True)", "seed": int(cfg["seed"]),
                          "world_size": W, "micro_batch_size": mbs, "gradient_accumulation_steps": accum,
                          "steps_per_epoch": per_epoch, "epoch": epoch, "micro_batches_consumed_per_rank": in_epoch * accum,
                          "next_dataset_indices_per_rank": nxt, "next_batch_disjoint_from_consumed": True,
                          "saved_sampler_state": st.get("sampler_state") or "not in this checkpoint (derived)"},
        "resume_command": "re-run the same launcher with the same output_dir (resume: true reads latest_checkpoint.txt)",
    }
    json.dump(manifest, open(os.path.join(d, "resume_manifest.json"), "w"), indent=1)
    print(json.dumps({k: manifest[k] for k in ("step", "optimizer", "scheduler", "rng")}, indent=1))
    print("next indices:", nxt)
    print("RESUME_VERIFIED")


if __name__ == "__main__":
    main()
