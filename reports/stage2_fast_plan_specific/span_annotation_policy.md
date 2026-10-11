# Span annotation policy (frozen before any evaluation)

Code: `later/scripts/data/plan_span_rules.py`. The exact lists (364 generic 3-grams, method lexicon, generic lead-ins)
are saved in `span_rules_frozen.json`; P1 asserts that it uses the identical lists.

**Priority per character:** subgoal_specific > method_specific > generic_template > unknown.

| label | rule | evidence |
|---|---|---|
| generic_template | header `I need to make a step-by-step plan.`; line-initial `N.` numbering; whitespace; word 3-grams in ≥1% of the 46,510 **train** plans (math stripped, method terms excluded); generic lead-in verbs (identify, determine, calculate, use, find, …) | frequency on train only |
| method_specific | whole-word match against a fixed lexicon of named techniques and operations (substitution, factorization, induction, parity, telescoping, inclusion-exclusion, modular, Vieta, derivative, law of cosines, …) | direct textual mention |
| subgoal_specific | LaTeX math spans (`$…$`, `$$…$$`, `\(…\)`, `\[…\]`) with ≥3 non-space characters, and standalone numbers with ≥2 digits outside math | problem-specific mathematical objects (**proxy**, not human-annotated subgoals) |
| unknown | everything else | — |

**Token mapping.** Qwen tokenizer offsets on the exact Stage-1 response string; token ids are asserted equal to the
PlanRec targets. A token takes the highest-priority label among its non-space characters. Whitespace-only tokens
count as generic.

**Quality check** (24 random examples in `p0_examples.md`, inspected manually):
- Precision of method and subgoal spans is good; marked items are real techniques and problem objects.
- Recall is low. Examples: "inclusion and exclusion" is missed (the lexicon has the hyphenated form only),
  1-digit numbers outside math are missed, and many problem-specific descriptive words fall into unknown (40% of
  tokens).
- Some lexicon entries are topic words rather than methods (probability, maximum, ratio).

The rules were **not** changed after inspection or after seeing P1 results.

**Label confidence: MEDIUM-LOW.** Specific spans are a high-precision, low-recall proxy. They are reliable enough to
test conditionality on the tokens they cover, but not to give a precise "strategy ratio".
