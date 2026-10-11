"""Mini Bridge pilot, data step 2: build verified dual-plan pairs from the Stage-1 samples.
Candidates: same question, both samples correct (math_verify vs reference answer), plans differ in method-lexicon terms,
and each plan's own method terms occur more often in its own execution than in the other execution (auto 'follows plan'
check). The accepted list below is the result of a manual read of all 89 auto-passing candidates (Claude, 2026-10-11):
rejected pairs had the same method in different words, a mathematically wrong plan step, or an unjustified 'known
result'. Split by question; held-out = the 8 pairs with the clearest method contrast."""
import json, re, sys
S = {(s["record_id"], s["sample"]): s for s in map(json.loads, open(sys.argv[1]))}
C = {c["record_id"]: c for c in json.load(open(sys.argv[2]))}
out = sys.argv[3]
HELDOUT = {  # record_id prefix -> short manual method labels (A | B)
    "s1val-00336": ("Lagrange multipliers", "Cauchy-Schwarz inequality"),
    "s1val-00574": ("difference-of-squares factoring of 65", "testing small n + modular arithmetic"),
    "s1val-00569": ("discriminant of a quadratic in x", "derivative / critical points"),
    "s1val-00507": ("differentiate F to a density + double integral", "inclusion-exclusion on the CDF"),
    "s1val-00479": ("factor into linear x quadratic", "Vieta's formula on the expanded cubic"),
    "s1val-00309": ("Euler's theorem (phi(9)=6)", "order of 4 mod 9 is 3"),
    "s1val-00305": ("substitute y=6/x + quadratic formula", "factor xy(x+y)+(x+y), then x^2+y^2 identity"),
    "s1val-00486": ("sum-of-cosines-in-AP identity", "7th roots of unity"),
}
TRAIN = ["s1val-00645", "s1val-00359", "s1val-00499", "s1val-00689", "s1val-00472", "s1val-00422", "s1val-00683",
         "s1val-00626", "s1val-00496", "s1val-00344", "s1val-00660", "s1val-00368", "s1val-00535", "s1val-00491",
         "s1val-00653", "s1val-00281", "s1val-00439", "s1val-00498", "s1val-00313", "s1val-00471"]
byp = {k.rsplit("-", 1)[0]: k for k in C}
def rec(pfx, labels=None):
    c = C[byp[pfx]]
    a, b = S[(c["record_id"], c["sA"])], S[(c["record_id"], c["sB"])]
    assert a["correct"] and b["correct"] and a["question"] == b["question"]
    return {"id": c["record_id"], "question": a["question"], "plan_A": a["plan"], "execution_A": a["execution"],
            "plan_B": b["plan"], "execution_B": b["execution"], "answer": a["gold_answer"], "pred_A": a["pred"],
            "pred_B": b["pred"], "verified_A": True, "verified_B": True, "sample_A": a["sample"], "sample_B": b["sample"],
            "method_terms_only_A": c["onlyA"], "method_terms_only_B": c["onlyB"],
            "method_label_A": labels[0] if labels else None, "method_label_B": labels[1] if labels else None,
            "verification": "answers: math_verify vs reference; method difference + plan-following: lexicon check + manual read"}
with open(f"{out}/train_pairs.jsonl", "w") as f:
    for p in TRAIN: f.write(json.dumps(rec(p)) + "\n")
with open(f"{out}/heldout_pairs.jsonl", "w") as f:
    for p, lab in HELDOUT.items(): f.write(json.dumps(rec(p, lab)) + "\n")
tr = {json.loads(l)["question"] for l in open(f"{out}/train_pairs.jsonl")}
ho = {json.loads(l)["question"] for l in open(f"{out}/heldout_pairs.jsonl")}
assert not tr & ho and len(tr) == 20 and len(ho) == 8
print("train", len(tr), "heldout", len(ho))
