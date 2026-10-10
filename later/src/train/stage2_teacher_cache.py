"""Loader for the offline Stage-2 teacher boundary-hidden cache (see later/scripts/data/cache_plan_boundary_hidden.py)."""
import json
import os

import torch


class TeacherBoundaryCache:
    """Loads all shards to CPU; lookup by record_id. Validates the manifest against the run's expectations."""

    def __init__(self, cache_dir: str, split: str, expect: dict):
        self.manifest = json.load(open(os.path.join(cache_dir, "manifest.json")))
        for k, v in expect.items():
            if self.manifest.get(k) != v:
                raise RuntimeError(f"teacher cache mismatch on {k}: cache={self.manifest.get(k)} expected={v}")
        self.index, chunks = {}, []
        off = 0
        for sh in self.manifest["splits"][split]["shards"]:
            blob = torch.load(os.path.join(cache_dir, sh["file"]))
            for i, rid in enumerate(blob["record_ids"]):
                self.index[rid] = off + i
            chunks.append(blob["hidden"])
            off += len(blob["record_ids"])
        self.hidden = torch.cat(chunks)
        self.layer_ids = list(self.manifest["layer_ids"])

    def get(self, record_ids, layer_ids, device):
        missing = [r for r in record_ids if r not in self.index]
        if missing:
            raise KeyError(f"teacher cache has no target for {missing[:3]} (+{max(0, len(missing) - 3)})")
        rows = torch.tensor([self.index[r] for r in record_ids])
        cols = torch.tensor([self.layer_ids.index(l) for l in layer_ids])
        h = self.hidden[rows][:, cols].to(device)  # [B, n_sel, d]
        return {l: h[:, j] for j, l in enumerate(layer_ids)}




class ExecKLTeacherCache:
    """Frozen-teacher execution-token top-k + tail cache (see later/scripts/data/cache_exec_teacher_topk.py)."""

    def __init__(self, cache_dir: str, split: str, expect: dict):
        self.manifest = json.load(open(os.path.join(cache_dir, "manifest.json")))
        for k, v in expect.items():
            if self.manifest.get(k) != v:
                raise RuntimeError(f"exec-KL cache mismatch on {k}: cache={self.manifest.get(k)} expected={v}")
        self.data = torch.load(os.path.join(cache_dir, f"{split}.pt"))
        self.split = split

    def get(self, record_id: str):
        if record_id not in self.data:
            raise KeyError(f"exec-KL cache ({self.split}) has no entry for {record_id}")
        return self.data[record_id]
