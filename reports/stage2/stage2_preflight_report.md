# Stage 2 preflight report (P0)

Date 2026-10-10 · node e4-1 (Slurm 48894985, 2 × NVIDIA B200, driver 590.48.01)

## Code
| | |
|---|---|
| LaTER | `https://github.com/TioeAre/LaTER` @ **6df72f7a26d46c7fc077d308e1b0765e134fecb9** (2026-07-06) |
| CODI (reference only) | `https://github.com/zhenyi4/codi` @ 2c2314662c63e9f482ebc46614ffe9af17a241e5 |
| Env | `~/.conda/envs/later-b200`: python 3.10, torch 2.8.0+cu128, transformers 5.12.1 (LaTER's pin), accelerate 1.15.0, flash-attn 2.8.3, peft 0.18.1. No deepspeed / vllm (not needed: DDP over 2 GPUs, full model per GPU). |

## Stage-1 checkpoint (start point, not retrained)
- `/scratch/hz33227/e2c/Explore-Execute-Chain/outputs/e2c_qwen3_4b_sft/best_hf` = **Qwen3-4B-E2C-SFT-only step 800** (full bf16 checkpoint, not LoRA; also `zhaohq/Qwen3-4B-E2C-SFT-only@main`). Model size: 4B (36 layers, d = 2560), not 14B.
- Checkpoint hash (config + index + first 64 MB of each shard): `d6fcd102…eae350f`; tokenizer.json sha256 `aeb13307…`; chat-template sha256 `a55ee1b1…`.
- Tokenizer unchanged from Qwen3-4B (vocab 151,936). **No LaTER special tokens are added** (see boundary section), so no embedding resize.

## Stage-1 data actually used for training (traced from the Stage-1 train log + run_manifest)
| split | file | sha256 | rows |
|---|---|---|---|
| train | `…/data/processed/sft/e2c-sft-train-fullfit.parquet` | `1f02662a…1e88a` | 49,684 |
| val | `…/data/processed/sft/e2c-sft-val-fullfit.parquet` | `14a403f8…f517c` | 2,611 |

Columns: `question`, `solution`, `solution_eos` (= `solution + "<|im_end|>"`, the Stage-1 response key).

## Stage-1 template and separator
- Prompt: `tokenizer.apply_chat_template([{user: Q}], add_generation_prompt=True)` (no `enable_thinking` kwarg) = `<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n` (identical to LaTER's `render_prompt_only`).
- Target: `<EXPLORATION>\n{P}\n</EXPLORATION><EXECUTION>\n{E}</EXECUTION><|im_end|>`. The answer A is inside E (last `\boxed{}`), as in Stage 1. No `<think>`.
- **Plan→Execution boundary = the existing separator `</EXPLORATION><EXECUTION>\n`, last token `>\n` (id 397).** It tokenizes to the same 9 tokens `</ EX PL OR ATION >< EXEC UTION >\n` in the teacher prefix and after the student's latents. The latent seed is the last token of `<EXPLORATION>\n`. **No new token is introduced**; LaTER's `<latent_think>/<think>` tokens are not used, because the Stage-1 teacher never saw them.

## Stage 1 → Stage 2 records (`stage1_to_stage2_data_audit.json`)
| | train | val |
|---|---|---|
| source rows | 49,684 | 2,611 |
| parsed (exact structure, byte-exact reassembly) | 49,321 | 2,597 |
| rejected: no parsable `\boxed` answer | 363 | 14 |
| rejected: plan restates the final answer (standalone expression, ≥4 chars, not in Q) | 2,542 | 127 |
| rejected: plan contains `\boxed` | 21 | 2 |
| rejected: overlap with an eval benchmark | 301 | 19 |
| rejected: question also in train | – | 7 |
| **accepted** | **46,510** | **2,446** |

- **Eval overlap** (normalized exact match, else any shared 13-word n-gram, against AIME24/25, AMC23, GSM8K, MATH-500, Minerva, OlympiadBench): per-record hits are olympiad_bench 179, aime24 99, math500 90, aime25 83, amc23 14. **Stage 1 itself was trained on these 320 records**, so any AIME24/25 / MATH-500 number for the Stage-1 or Stage-2 checkpoints is contaminated.
- Unique questions: train 46,420 (90 rows repeat a question, all within train); val 2,444; 0 shared between train and val.
- The data has no gold-answer column; A is the Stage-1 target's own answer and cannot be verified.
- 67% of plan steps are restated verbatim in E (E2C format). These lines are part of the execution target the student must produce from latents.
- All 48,956 records build the three views without error (`data/view_build_check.json`). Mean lengths at L = 8: teacher prefix 289, student latent 1,010, explicit 1,177 tokens; max explicit 2,910 (≤ 4,096, nothing truncated).

## Risks found before implementation
1. LaTER `LatentQwenConfig.latent_embed_builder_no_grad=True`, and `train.py` never copied the YAML value into the config. Fixed (2-line change, baseline default unchanged). Verified by test: with `no_grad=True` the projector gradient is **exactly 0** (the projector only runs inside the builder).
2. **Upstream LaTER `train.py` imports `later.src.train.trainer_latent_cp`, which does not exist at this commit**, so the original LaTER training entry fails at import regardless of our changes. Our Stage-2 entry (`train_stage2.py`) does not depend on it. Not patched (it would change the baseline).
3. LaTER's `_forward_global_stateless_teacher_forcing` returns no hidden states (`loss_hidden_states=None`). Added a separate Stage-2 forward with hook-based boundary capture.
4. A forward hook that indexes inside a gradient-checkpointed layer breaks recomputation (52 vs 50 saved tensors). Fixed by gathering after the forward. Tested with GC on.
5. Teacher/student RoPE positions at the boundary differ (teacher ≈ 289, student ≈ prompt + 6 + L + 9). This is expected and logged, not adjusted.
