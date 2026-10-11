"""Frozen, rule-based character labels for E2C plan text (fast plan-specificity diagnostic).

Priority per character: subgoal_specific > method_specific > generic_template > unknown.
  generic_template : header line 'I need to make a step-by-step plan.', line-initial step numbering, whitespace,
                     and word 3-grams with document frequency >= GENERIC_DF over the FULL TRAIN plans (built once from
                     train only; never from val or from model results), plus generic action lead-ins listed below.
  method_specific  : explicit named techniques / key operations from a fixed lexicon (direct textual evidence only).
  subgoal_specific : problem-specific mathematical objects: LaTeX math spans ($...$, $$...$$, \\(...\\), \\[...\\])
                     with >= 3 non-space characters inside, and standalone numbers outside math with >= 2 digits.
                     (Proxy for "key subgoal / equation / object" — not human-annotated subgoals.)
  unknown          : everything else (ordinary words that are neither listed generic n-grams nor method terms).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, List

HEADER = "I need to make a step-by-step plan."
GENERIC_DF = 0.01          # frozen before any evaluation
METHOD_LEXICON = [
    "factoriz", "factorise", "factorize", "factor ", "factoring", "prime factor", "substitut", "induction", "case analysis",
    "cases", "case ", "modular", "modulo", "mod ", "congruen", "quadratic formula", "completing the square",
    "complete the square", "vieta", "am-gm", "am–gm", "cauchy", "inequality", "inclusion-exclusion", "pigeonhole",
    "derivative", "differentiat", "integrat", "antiderivative", "law of cosines", "law of sines", "pythagor",
    "similar triangle", "binomial theorem", "binomial coefficient", "telescop", "recurrence", "recursion", "recursive",
    "logarithm", "symmetry", "symmetric", "coordinate", "parametri", "complementary counting", "complement",
    "geometric series", "arithmetic sequence", "arithmetic progression", "geometric sequence", "geometric progression",
    "euclidean algorithm", "gcd", "lcm", "chinese remainder", "fermat", "euler", "generating function", "contradiction",
    "dynamic programming", "permutation", "combination", "stars and bars", "probability", "expected value",
    "vector", "dot product", "cross product", "matrix", "determinant", "eigen", "trigonometric identit", "double angle",
    "sum-to-product", "product-to-sum", "angle bisector", "power of a point", "shoelace", "heron", "area formula",
    "volume formula", "distance formula", "midpoint formula", "slope", "tangent line", "limit", "l'hopital",
    "l'hôpital", "taylor", "series expansion", "partial fraction", "rationaliz", "conjugate", "discriminant",
    "polynomial division", "synthetic division", "remainder theorem", "rational root", "graph", "monoton",
    "extremum", "maximum", "minimum", "optimiz", "lagrange", "parity", "divisib", "floor function", "ceiling",
    "absolute value", "change of base", "exponent rule", "logarithm rule", "law of exponents", "system of equations",
    "elimination", "cross-multipl", "proportion", "ratio", "percent", "unit conversion", "counting principle",
    "multiplication principle", "addition principle", "bijection", "invariant", "extremal", "mathematical induction",
]
GENERIC_LEADINS = ["identify", "determine", "calculate", "compute", "find", "use", "apply", "recognize", "recall",
                   "express", "simplify", "solve", "evaluate", "substitute the", "verify", "check", "conclude",
                   "understand", "analyze", "consider", "note that", "set up", "write", "the problem", "given"]


def _words(s):
    return re.findall(r"[a-z]+", s.lower())


def build_generic_ngrams(train_plans: List[str]) -> List[str]:
    df = Counter()
    for p in train_plans:
        body = "\n".join(l for l in p.split("\n") if l.strip() != HEADER)
        body = re.sub(r"\$[^$]*\$", " ", body)
        w = _words(body)
        df.update({" ".join(w[i:i + 3]) for i in range(len(w) - 2)})
    n = len(train_plans)
    meth = [m.strip() for m in METHOD_LEXICON]
    return sorted(g for g, c in df.items() if c / n >= GENERIC_DF and not any(m in g for m in meth))


def label_chars(plan: str, generic_ngrams: List[str]) -> List[str]:
    lab = ["unknown"] * len(plan)
    low = plan.lower()

    def mark(a, b, tag):
        for c in range(a, b):
            lab[c] = tag
    # generic
    for c, ch in enumerate(plan):
        if ch.isspace():
            lab[c] = "generic_template"
    pos = 0
    for i, line in enumerate(plan.split("\n")):
        if i == 0 and line.strip() == HEADER:
            mark(pos, pos + len(line), "generic_template")
        m = re.match(r"\s*\d+\.\s*", line)
        if m:
            mark(pos, pos + m.end(), "generic_template")
        pos += len(line) + 1
    for g in generic_ngrams:
        pat = r"\b" + r"[^a-z$]+".join(map(re.escape, g.split())) + r"\b"
        for m in re.finditer(pat, low):
            mark(m.start(), m.end(), "generic_template")
    for g in GENERIC_LEADINS:
        for m in re.finditer(r"\b" + re.escape(g) + r"\b", low):
            mark(m.start(), m.end(), "generic_template")
    # method (overrides generic)
    for t in METHOD_LEXICON:
        t2 = t.strip()
        for m in re.finditer(r"\b" + re.escape(t2) + (r"\b" if t.endswith(" ") else r"[a-z\-]*"), low):
            mark(m.start(), m.end(), "method_specific")
    # subgoal (overrides all)
    math_spans = []
    for m in re.finditer(r"\$\$[^$]+\$\$|\$[^$]+\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", plan, flags=re.S):
        inner = re.sub(r"^\$+|\$+$|^\\[\(\[]|\\[\)\]]$", "", m.group(0))
        if len(re.sub(r"\s", "", inner)) >= 3:
            mark(m.start(), m.end(), "subgoal_specific")
        math_spans.append((m.start(), m.end()))
    for m in re.finditer(r"(?<![\w.])\d[\d,.]*\d(?![\w])", plan):
        if not any(a <= m.start() < b for a, b in math_spans):
            mark(m.start(), m.end(), "subgoal_specific")
    return lab


def spans_of(lab: List[str], tag: str):
    out, s = [], None
    for i, t in enumerate(lab + ["#"]):
        if t == tag and s is None:
            s = i
        elif t != tag and s is not None:
            out.append((s, i))
            s = None
    return out


def save_policy(path: str, generic_ngrams: List[str]) -> None:
    json.dump({"generic_df_threshold": GENERIC_DF, "n_generic_ngrams": len(generic_ngrams), "generic_ngrams": generic_ngrams,
               "method_lexicon": METHOD_LEXICON, "generic_leadins": GENERIC_LEADINS}, open(path, "w"), indent=1)
