"""OUR Stage 2 (deterministic latent plan distillation) dataset: three views of one Stage-1 record.

Stage-1 sample (exact training tokenization, prompt and response tokenized separately then concatenated):
    prompt_ids  = tok(apply_chat_template([{user: Q}], add_generation_prompt=True))
    resp_ids    = tok("<EXPLORATION>\\n" P "\\n</EXPLORATION><EXECUTION>\\n" E "</EXECUTION><|im_end|>")

All three views are slices of these ids (no re-tokenization of fragments), so every shared token is identical
to what the Stage-1 checkpoint was trained on:

  teacher_prefix   = prompt | <EXPLORATION>\\n P \\n</EXPLORATION><EXECUTION>\\n          (ends at boundary b_T)
  student_latent   = prompt | <EXPLORATION>\\n [z_1..z_L] </EXPLORATION><EXECUTION>\\n | E </EXECUTION><|im_end|>
                                ^ latent_start (seed hidden)                    ^ boundary b_S
  student_explicit = prompt | <EXPLORATION>\\n P \\n</EXPLORATION><EXECUTION>\\n E </EXECUTION><|im_end|>
                                                                              ^ boundary b_X (logged only)

The boundary token is the Stage-1 separator's last token '>\\n' (id 397) closing '<EXECUTION>'; the 9-token
suffix '</ EX PL OR ATION >< EXEC UTION >\\n' is identical in teacher and student. No new special token.
Labels are token-aligned (HF convention, -100 = ignored; shift happens in the loss):
  student_latent  : E + A + closing '</EXECUTION><|im_end|>' supervised; prompt, '<EXPLORATION>\\n', latent slots
                    and the forced separator are not targets (no implicit halting objective).
  student_explicit: the whole response (P + E + A + tags) supervised, exactly like Stage 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

PLAN_OPEN = "<EXPLORATION>\n"
SEPARATOR = "\n</EXPLORATION><EXECUTION>\n"
EXEC_CLOSE = "</EXECUTION>"
EOS = "<|im_end|>"
STUDENT_SUFFIX = "</EXPLORATION><EXECUTION>\n"  # forced after the latents; tokenizes to the teacher's 9 tail tokens
BOUNDARY_RULE_VERSION = "e2c-v1: boundary = last token ('>\\n', id 397) of '</EXPLORATION><EXECUTION>\\n'"


@dataclass
class Stage2Tokens:
    plan_open_ids: List[int]
    suffix_ids: List[int]
    boundary_id: int
    pad_id: int


def stage2_tokens(tokenizer: Any) -> Stage2Tokens:
    plan_open = tokenizer.encode(PLAN_OPEN, add_special_tokens=False)
    suffix = tokenizer.encode(STUDENT_SUFFIX, add_special_tokens=False)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.convert_tokens_to_ids("<|endoftext|>")
    return Stage2Tokens(plan_open_ids=plan_open, suffix_ids=suffix, boundary_id=suffix[-1], pad_id=int(pad))


def render_prompt(tokenizer: Any, question: str) -> str:
    return tokenizer.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True,
                                         tokenize=False)


class Stage2RecordError(ValueError):
    pass


KL_MASK_RULE = ("e2c-kl-v1: execution targets with segment E (not A, not closing tags), excluding tokens on lines "
                "that restate a plan line verbatim; first `kl_budget` such targets, indexed by execution offset")


def build_views(tokenizer: Any, row: Dict[str, Any], latent_steps: int, toks: Stage2Tokens,
                max_length: int, kl_budget: int = 128) -> Dict[str, Any]:
    q, plan, execution = str(row["question"]), str(row["plan"]), str(row["execution"])
    response = PLAN_OPEN + plan + SEPARATOR + execution + EXEC_CLOSE + EOS
    if response != str(row["stage1_solution"]) + EOS:
        raise Stage2RecordError("response_not_equal_stage1_target")
    prompt_ids = tokenizer.encode(render_prompt(tokenizer, q), add_special_tokens=False)
    enc = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    resp_ids, offs = list(enc["input_ids"]), list(enc["offset_mapping"])
    full = prompt_ids + resp_ids
    P = len(prompt_ids)
    n_open = len(toks.plan_open_ids)
    if resp_ids[:n_open] != toks.plan_open_ids:
        raise Stage2RecordError("plan_open_tokenization_differs")
    # locate the separator tail: the 9 suffix tokens must appear exactly once, right before the execution
    sfx = toks.suffix_ids
    hits = [i for i in range(n_open, len(resp_ids) - len(sfx) + 1) if resp_ids[i:i + len(sfx)] == sfx]
    if len(hits) != 1:
        raise Stage2RecordError("separator_suffix_not_unique")
    sep_start = hits[0]                      # index in resp of '</'
    exec_start = sep_start + len(sfx)        # first execution token (resp index)
    exec_char0 = len(PLAN_OPEN) + len(plan) + len(SEPARATOR)
    if offs[exec_start][0] != exec_char0:
        raise Stage2RecordError("execution_does_not_start_on_token_boundary")
    a_char0 = exec_char0 + int(row["answer_char_start_in_execution"])
    close_char0 = exec_char0 + len(execution)
    n_resp = len(resp_ids)
    seg = []  # per response token: P / E / A / C (closing tags + EOS) / T (plan-side tags)
    for i, (a, b) in enumerate(offs):
        if i < n_open:
            seg.append("T")
        elif i < sep_start:
            seg.append("P")
        elif i < exec_start:
            seg.append("T")
        elif b <= close_char0:
            seg.append("A" if a >= a_char0 else "E")
        elif a < close_char0:          # token straddles last execution char and '</' (e.g. '.</')
            seg.append("E")
        else:
            seg.append("C")
    # monitoring only (labels unchanged): execution tokens on lines that restate a plan line verbatim
    plan_lines = {l.strip() for l in plan.split("\n") if len(l.strip()) > 3}
    rep_spans, pos = [], exec_char0
    for line in execution.split("\n"):
        if line.strip() in plan_lines:
            rep_spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    rep_tok = [i >= exec_start and any(a < e and b > st for st, e in rep_spans) for i, (a, b) in enumerate(offs)]

    # teacher prefix: Stage-1 ids up to and including the boundary token
    b_T = P + exec_start - 1
    teacher_ids = full[: b_T + 1]
    assert teacher_ids[-1] == toks.boundary_id

    # student latent path
    latent_start = P + n_open - 1
    exec_tail = resp_ids[exec_start:]
    student_ids = full[: P + n_open] + [toks.pad_id] * latent_steps + sfx + exec_tail
    latent_positions = list(range(P + n_open, P + n_open + latent_steps))
    b_S = P + n_open + latent_steps + len(sfx) - 1
    assert student_ids[b_S] == toks.boundary_id
    s_labels = [-100] * (b_S + 1) + exec_tail
    s_seg = ["-"] * (b_S + 1) + seg[exec_start:]
    s_rep = [False] * (b_S + 1) + rep_tok[exec_start:]

    # student explicit path = the Stage-1 sample itself
    x_labels = [-100] * P + resp_ids
    x_seg = ["-"] * P + seg
    x_rep = [False] * P + rep_tok
    b_X = b_T

    if len(full) > max_length or len(student_ids) > max_length:
        raise Stage2RecordError("exceeds_max_length")
    return {
        "record_id": str(row["record_id"]),
        "teacher_ids": teacher_ids, "teacher_boundary": b_T,
        "student_ids": student_ids, "student_labels": s_labels, "student_seg": "".join(s_seg),
        "latent_start": latent_start, "latent_positions": latent_positions, "student_boundary": b_S,
        "explicit_ids": full, "explicit_labels": x_labels, "explicit_seg": "".join(x_seg), "explicit_boundary": b_X,
        "prompt_len": P,
        "student_rep": s_rep, "explicit_rep": x_rep,
        # execution offsets j (0 = first execution token = target predicted from the boundary) used by the
        # optional execution-KL pilot; identical meaning in the student and explicit/teacher sequences
        "kl_offsets": [j for j, (sg, rp) in enumerate(zip(seg[exec_start:], rep_tok[exec_start:]))
                       if sg == "E" and not rp][:kl_budget],
        "n_exec_targets": len(exec_tail),
    }


class Stage2PlanDataset(Dataset):
    def __init__(self, parquet: str, tokenizer: Any, latent_steps: int, max_length: int = 4096,
                 limit: int | None = None):
        self.frame = pd.read_parquet(parquet)
        if limit:
            self.frame = self.frame.iloc[:limit].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.latent_steps = int(latent_steps)
        self.max_length = int(max_length)
        self.toks = stage2_tokens(tokenizer)
        assert self.latent_steps >= 1

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return build_views(self.tokenizer, self.frame.iloc[idx].to_dict(), self.latent_steps, self.toks,
                           self.max_length)


def _pad(seqs: Sequence[Sequence[int]], fill: int) -> torch.Tensor:
    n = max(len(s) for s in seqs)
    return torch.tensor([list(s) + [fill] * (n - len(s)) for s in seqs], dtype=torch.long)


def _mask(seqs: Sequence[Sequence[int]]) -> torch.Tensor:
    n = max(len(s) for s in seqs)
    return torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs], dtype=torch.long)


def _seg_mask(segs: Sequence[str], n: int, letters: str) -> torch.Tensor:
    return torch.tensor([[c in letters for c in s] + [False] * (n - len(s)) for s in segs], dtype=torch.bool)


class Stage2Collator:
    """Right padding. position_ids = arange over real tokens (same as Stage-1 training and generation)."""

    def __init__(self, pad_id: int):
        self.pad_id = int(pad_id)

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"record_ids": [b["record_id"] for b in batch]}
        for view, ids_key, lab_key in [("student", "student_ids", "student_labels"),
                                       ("explicit", "explicit_ids", "explicit_labels")]:
            ids = [b[ids_key] for b in batch]
            n = max(len(s) for s in ids)
            out[f"{view}_input_ids"] = _pad(ids, self.pad_id)
            out[f"{view}_attention_mask"] = _mask(ids)
            out[f"{view}_position_ids"] = torch.arange(n).unsqueeze(0).expand(len(batch), -1).clone()
            out[f"{view}_labels"] = _pad([b[lab_key] for b in batch], -100)
            segs = [b[f"{view}_seg"] for b in batch]
            for name, letters in [("P", "P"), ("E", "E"), ("A", "A"), ("C", "C")]:
                out[f"{view}_seg_{name}"] = _seg_mask(segs, n, letters)
            reps = [b[f"{view}_rep"] for b in batch]   # monitoring only: plan-restatement tokens
            out[f"{view}_seg_R"] = torch.tensor([list(r) + [False] * (n - len(r)) for r in reps], dtype=torch.bool)
        out["kl_offsets"] = [b["kl_offsets"] for b in batch]
        out["latent_start_positions"] = torch.tensor([b["latent_start"] for b in batch], dtype=torch.long)
        out["latent_positions"] = torch.tensor([b["latent_positions"] for b in batch], dtype=torch.long)
        out["student_boundary"] = torch.tensor([b["student_boundary"] for b in batch], dtype=torch.long)
        out["explicit_boundary"] = torch.tensor([b["explicit_boundary"] for b in batch], dtype=torch.long)
        t_ids = [b["teacher_ids"] for b in batch]
        out["teacher_input_ids"] = _pad(t_ids, self.pad_id)
        out["teacher_attention_mask"] = _mask(t_ids)
        out["teacher_boundary"] = torch.tensor([b["teacher_boundary"] for b in batch], dtype=torch.long)
        return out
