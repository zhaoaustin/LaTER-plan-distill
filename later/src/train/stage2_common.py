"""Shared loaders / hashing for OUR Stage 2 (plan distillation). Teacher and student use the same model class
and the same boundary-capture code, so layer semantics are identical by construction."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict

import torch
from transformers import AutoConfig, AutoTokenizer

from later.src.train.modeling_latent import (LatentQwenConfig, LatentQwenForCausalLM,
                                             register_latent_qwen3_with_transformers)


def file_sha256(path: str, limit_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        read = 0
        while b := f.read(1 << 24):
            h.update(b)
            read += len(b)
            if limit_bytes and read >= limit_bytes:
                break
    return h.hexdigest()


def checkpoint_hash(path: str) -> str:
    """Hash of config + safetensors index + first 64MB of every weight shard (cheap but changes on any retrain)."""
    h = hashlib.sha256()
    for name in sorted(os.listdir(path)):
        p = os.path.join(path, name)
        if name.endswith(".safetensors"):
            h.update(name.encode() + file_sha256(p, 64 << 20).encode())
        elif name in ("config.json", "model.safetensors.index.json"):
            h.update(name.encode() + file_sha256(p).encode())
    return h.hexdigest()


def tokenizer_hashes(tokenizer: Any, path: str) -> Dict[str, str]:
    tj = os.path.join(path, "tokenizer.json")
    tmpl = tokenizer.chat_template or ""
    return {"tokenizer_hash": file_sha256(tj) if os.path.exists(tj) else "n/a",
            "chat_template_hash": hashlib.sha256(tmpl.encode()).hexdigest()}


def load_tokenizer(path: str) -> Any:
    return AutoTokenizer.from_pretrained(path)


def load_stage2_model(path: str, *, latent_embed_builder_no_grad: bool, latent_projector_dropout: float,
                      dtype: torch.dtype, attn_implementation: str = "sdpa",
                      latent_output_rms: Any = None) -> LatentQwenForCausalLM:
    """Wrap a Stage-1 Qwen3 checkpoint (or a saved Stage-2 checkpoint) as LatentQwenForCausalLM.

    latent_output_rms: None (LaTER projector as-is), "token_embedding" (rescale projector outputs to the per-dim
    RMS of the input-embedding matrix, measured once on the Stage-1 weights) or a float. A saved Stage-2
    checkpoint keeps the value stored in its config.
    """
    register_latent_qwen3_with_transformers()
    base = AutoConfig.from_pretrained(path)
    cfg = base if isinstance(base, LatentQwenConfig) else LatentQwenConfig.from_dict(base.to_dict())
    cfg.latent_embed_builder_no_grad = bool(latent_embed_builder_no_grad)
    cfg.latent_projector_dropout = float(latent_projector_dropout)
    cfg.global_stateless_forward = True
    model = LatentQwenForCausalLM.from_pretrained(path, config=cfg, dtype=dtype,
                                                  attn_implementation=attn_implementation)
    # LatentProjector builds nn.Dropout at construction time from the config value
    for m in model.latent_projector.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = float(latent_projector_dropout)
    stored = getattr(model.config, "latent_projector_output_rms", None)
    if stored:
        model.latent_projector.output_rms = float(stored)
    elif latent_output_rms == "token_embedding":
        w = model.get_input_embeddings().weight.detach().float()
        value = float(w.pow(2).mean().sqrt())
        model.latent_projector.output_rms = value
        model.config.latent_projector_output_rms = value
    elif latent_output_rms not in (None, False, "none"):
        model.latent_projector.output_rms = float(latent_output_rms)
        model.config.latent_projector_output_rms = float(latent_output_rms)
    return model


def dump_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=str)
