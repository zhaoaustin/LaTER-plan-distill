# Test 1: explicit-mode 0/50, forgetting or evaluation bug?

**Verdict: EXPLICIT_FORGETTING_SUPPORTED.** What is lost is mainly the E²C output format. Math is lost only in B.

## Setup
- **Problems:** the same 50 problems for every run (`clean_math_eval.parquet`: MATH-500, 13-gram clean against the
  full E²C SFT source).
- **Prompt:** Qwen3 chat template with the user question and `add_generation_prompt=True`; nothing appended.
- **Decoding:** greedy, max 3,072 new tokens, EOS `<|im_end|>` / `<|endoftext|>`.
- **Format tags:** `<EXPLORATION>`, `</EXPLORATION><EXECUTION>` and `</EXECUTION>` are ordinary BPE sequences, not
  special tokens. Detection is done on the decoded text.
- **Format-free math:** the last `\boxed{}` anywhere in the output, checked with math_verify.

| run (load / protocol) | format valid | accuracy (format parser) | **math, last \boxed anywhere** | answer parsed | EOS stop | mean tokens | categories |
|---|---|---|---|---|---|---|---|
| Stage 1, native HF / native SFT | 0.96 | 0.76 | 0.76 | 0.96 | 0.96 | 755 | ok 48, truncated 2 |
| Stage 1, wrapper / native SFT | 0.90 | 0.70 | 0.70 | 0.90 | 0.90 | 879 | ok 45, truncated 5 |
| Stage 1, wrapper / Stage-2 loop | 0.94 | 0.72 | 0.72 | 0.94 | 0.94 | 779 | ok 47, truncated 3 |
| A step 300, native / native SFT | **0.00** | 0.00 | **0.74** | 0.92 | 0.84 | 937 | no `<EXPLORATION>` 42, truncated 8 |
| A step 300, wrapper / Stage-2 loop | **0.00** | 0.00 | **0.76** | 0.96 | 0.92 | 714 | no `<EXPLORATION>` 46, truncated 4 |
| B step 300, native / native SFT | **0.00** | 0.00 | **0.56** | 0.96 | 0.56 | 1,598 | no `<EXPLORATION>` 28, truncated 22 |
| B step 300, wrapper / Stage-2 loop | **0.00** | 0.00 | **0.58** | 0.94 | 0.54 | 1,652 | no `<EXPLORATION>` 27, truncated 23 |

## Ruling out an interface bug
- **Stage 1 works under both loading paths and both protocols.**
  - Native vs wrapper next-token distribution at the end of all 50 prompts: top-1 agrees 50/50, mean KL 1.3e-6.
  - The first 10 greedy tokens are identical for 10/10 prompts.
  - Stage-1 wrapper + Stage-2 loop reproduces the pilot's step-0 explicit result exactly (0.72, mean length 778.94).
  - The 0.70–0.76 spread among the Stage-1 rows comes from fp32+autocast vs bf16 numerics over long greedy decodes.
- **A and B fail identically under the native HF path and protocol**, so the wrapper is not the cause. Native vs
  wrapper agreement on per-problem math correctness: A 94%, B 90%.

## What changed in A and B (same problem, paired)
- **First token.** Stage 1's top-1 first token is `<` (start of `<EXPLORATION>`) for 50/50 prompts.
  - A: top-1 is a newline or other scattered tokens; A writes execution-style steps directly.
  - B: top-1 is `I`; B writes "I need to make a step-by-step plan…" without the tag. 31/50 B outputs contain an
    early `Answer: $\boxed{…}$` line, and none contain `</EXPLORATION><EXECUTION>`.
- **Math, format-free, exact McNemar test:**
  - Stage 1 vs A: 31 both correct, 7 Stage-1 only, 6 A-only (p = 1.0). **No measurable loss of math for A.**
  - Stage 1 vs B: 14 Stage-1 only, 4 B-only (p = 0.031).
  - A vs B: 12 A-only, 3 B-only (p = 0.035).
  - **B loses math.**
- **B's extra failure is non-termination with repetition.** 22–23 of 50 outputs hit the token limit. In those
  outputs 87% of lines are repeats, vs 25% in B outputs that stopped. Even when B stops, its math accuracy is 0.54
  (A: 0.81).
- **Cause.** Neither A nor B supervises the explicit path. Pilot B's plan-reconstruction branch additionally trains
  the backbone to emit plan text after `[z] + <EXPLORATION>\n`, which is consistent with B's plan-first, untagged output.
  The mechanism for B's repetition is not established.

The format 0/50 does **not** mean 0% math: A keeps Stage-1-level accuracy once the format is ignored. Samples:
`samples.md`; every raw output: `per_example.jsonl`.
