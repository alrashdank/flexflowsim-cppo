#!/usr/bin/env python3
"""Claim-level checks on the manuscript.

The numeric harness (verify_ijms.py) reconstructs every table cell from the
archived records and asserts the formatted string appears in the text.  It
passed on the R2 manuscript while four defects stood, because those defects were
claims ABOUT the numbers rather than wrong numbers:

  - a quantifier ranging over a wider set than the measurement ("every policy in
    the study" against a figure measured on seven load-spreading policies);
  - a significance word contradicting the matching row of Table 3
    ("worse ... than random routing" against +1.99 [-1.05, +4.90], p = 0.20);
  - an argument whose conclusion did not follow from the range it quoted;
  - a stale duplicate of a corrected figure surviving in the introduction.

This script checks those classes.  It is deliberately noisy: every hit is
reported for a human to clear, not silently passed.

    python3 verify_claims.py IJMS_manuscript.md
"""

import json
import re
import sys
from pathlib import Path

TEXT = Path(sys.argv[1] if len(sys.argv) > 1 else "IJMS_manuscript.md").read_text()
LINES = TEXT.split("\n")

fails, warns = [], []


def fail(tag, msg):
    fails.append(f"[{tag}] {msg}")


def warn(tag, msg):
    warns.append(f"[{tag}] {msg}")


def locate(pattern, flags=re.I):
    for i, l in enumerate(LINES, 1):
        if re.search(pattern, l, flags):
            yield i, l.strip()


# --------------------------------------------------------------- 1 quantifiers
# Every universal quantifier attached to a measured quantity must name the set it
# ranges over, and that set must be one the study actually measured.
MEASURED_SETS = {
    "the seven load-spreading policies", "load-spreading policies",
    "test episodes", "seeds", "training episodes", "checkpoints",
    "one-sided configuration", "control seed", "validated",
}
QUANT = r"\b(?:every|all|any|each)\b"
# A universal is fine when it is RESTRICTED ("any policy whose mean slack ...")
# or when it states infeasibility rather than a measurement.  It is a defect when
# it attaches to something the study measured on a proper subset.
MEASURING = r"(\bmeasured\b|\bspans?\b|\bvaries\b|\bobserved\b|\branges\b)"

for i, l in locate(rf"{QUANT}\s+polic(?:y|ies)"):
    if re.search(rf"{QUANT}\s+polic(?:y|ies)\s+(?:whose|that|which|achieving)", l, re.I):
        continue                                   # restricted quantifier
    if re.search(r"infeasible|deployable|would be", l, re.I):
        continue
    ctx = " ".join(LINES[max(0, i - 3): i + 2]).lower()
    has_number = re.search(r"\d+(?:\.\d+)?\s*%|\$[\d,]+", ctx)
    if (re.search(MEASURING, ctx) and has_number
            and "load-spreading" not in ctx and "trained polic" not in ctx):
        fail("quantifier",
             f"L{i}: universal over policies attached to a measurement without naming the set -> {l[:95]}")

for i, l in locate(rf"{QUANT}\s+(?:policy|cell|seed|episode|configuration)\s+in\s+(?:the|this)\s+study"):
    fail("quantifier", f"L{i}: 'in the study' quantifier -> {l[:95]}")

# ------------------------------------------------------- 2 significance words
# A word asserting a difference must match the Table 3 row it refers to.
TABLE3 = {}
CELLRE = re.compile(r"([+−-][\d.]+)\s*\[([^\]]+)\]")


def _num(s):
    return float(s.replace("−", "-").replace("+", "").strip())


for l in LINES:
    m = re.match(r"\|\s*(electronics|bakery),\s*([\w ()]+?)\s*\|\s*([\w ()]+?)\s*\|(.*)$", l)
    if not m:
        continue
    tb, cell, comp, rest = m.groups()
    cols = [c.strip() for c in rest.split("|")]
    # rest is: dCPU | p | dsat | p |   -- parse each (estimate, p) pair separately
    pairs = []
    for j in range(0, len(cols) - 1):
        est = CELLRE.match(cols[j])
        if est and j + 1 < len(cols):
            lo, hi = est.group(2).split(",")
            pairs.append((("cpu" if not pairs else "sat"),
                          _num(est.group(1)), _num(lo) <= 0 <= _num(hi), cols[j + 1]))
    for metric, diff, spans_zero, p in pairs:
        TABLE3[(tb.strip(), cell.strip().lower(), comp.strip().lower(), metric)] = dict(
            diff=diff, spans_zero=spans_zero, p=p.replace("*", "").strip(),
            starred="*" in p)

ASSERTIVE = r"\b(worse than|better than|more expensive than|cheaper than|costing more per unit than|ahead of|below|above)\b"
STATELESS = ("random routing", "stateless routing", "roundrobin", "round-robin",
             "uniformrandom", "uniform-random")

for i, l in locate(ASSERTIVE):
    low = l.lower()
    if not any(s in low for s in STATELESS):
        continue
    if any(h in low for h in ("no better", "not distinguishable", "indistinguishable",
                             "no cheaper", "not separable", "neither resolvable",
                             "not below", "failure to detect")):
        continue
    warn("significance",
         f"L{i}: assertive comparison against stateless routing; check against Table 3 -> {l[:95]}")

for key, row in TABLE3.items():
    if row["spans_zero"] and row["starred"]:
        fail("table3", f"{key}: interval spans zero but entry is starred")
    try:
        pv = float(row["p"].replace("<", "").strip())
    except ValueError:
        continue
    if row["spans_zero"] and pv < 0.05:
        fail("table3", f"{key}: interval spans zero but p = {row['p']}")
    if not row["spans_zero"] and pv > 0.05:
        fail("table3", f"{key}: interval excludes zero but p = {row['p']}")

# ------------------------------------------- 3 duplicated quantities must agree
# Any percentage or dollar figure stated twice must be stated identically.
def spans(pat):
    return {m.group(0) for m in re.finditer(pat, TEXT)}

cost_spans = spans(r"(?:cost )?(?:varies by|spans) \d+(?:\.\d+)?%")
tp_spans = sorted({m.group(1) for m in re.finditer(
    r"throughput (?:varies by|spans|across the same policies spans) (\d+(?:\.\d+)?)%", TEXT)})
cs = sorted({m.group(1) for m in re.finditer(
    r"(?:episode )?cost (?:varies by|spans) (\d+(?:\.\d+)?)%", TEXT)})
if len(cs) > 1:
    fail("duplicate", f"cost span stated inconsistently: {cs}")
if len(tp_spans) > 1:
    fail("duplicate", f"throughput span stated inconsistently: {tp_spans}")

AGENT, SQ = 78.10118180724187, 73.25
# "SQ is X% cheaper than the agent"  -> (agent - SQ) / agent
# "the agent is X% more expensive than SQ" -> (agent - SQ) / SQ
IMPLIED = {"cheaper": (AGENT - SQ) / AGENT * 100,
           "more expensive": (AGENT - SQ) / SQ * 100}
for val, direction in re.findall(r"(\d+\.\d+)% (cheaper|more expensive)", TEXT):
    if abs(float(val) - IMPLIED[direction]) > 0.15:
        fail("percentage",
             f"'{val}% {direction}' does not match $78.10 vs $73.25 "
             f"(implies {IMPLIED[direction]:.2f}%)")

# ------------------------------------- 4 arguments quoting a range as a warrant
for i, l in locate(r"increment of .*? to .*?, so\b"):
    fail("argument", f"L{i}: conclusion drawn from a range rather than its mean -> {l[:95]}")

# --------------------------------------- 5 new figures must exist in r3_numbers
r3p = Path("/home/claude/repo/r3_numbers.json")
if r3p.exists():
    r3 = json.loads(r3p.read_text())
    checks = [
        ("0.0031", f"{r3['fill']['bakery']['mean_g_T']:.4f}"),
        ("0.0054", f"{r3['fill']['electronics']['mean_g_T']:.4f}"),
        ("12.5", f"{r3['fill']['bakery']['increment_per_episode']:.1f}"),
        ("21.6", f"{r3['fill']['electronics']['increment_per_episode']:.1f}"),
        ("6.30", f"{r3['fill']['bakery']['increment_needed_for_cap']:.2f}"),
        ("5.91", f"{r3['fill']['electronics']['increment_needed_for_cap']:.2f}"),
        ("1,581", f"{r3['fill']['bakery']['cap_at_episode']:,}"),
        ("915", f"{r3['fill']['electronics']['cap_at_episode']:,}"),
        ("84.8", f"{100*r3['symsat']['stochastic']['tp_satisfied']:.1f}"),
        ("82.0", f"{100*r3['symsat']['stochastic']['util_satisfied']:.1f}"),
        ("74.8", f"{100*r3['symsat']['stochastic']['joint_satisfied']:.1f}"),
    ]
    for in_text, from_data in checks:
        if in_text not in TEXT:
            fail("r3", f"{in_text!r} absent from manuscript (data says {from_data})")
        # 1,582 in prose vs 1581 ceiling: allow the off-by-one of rounding up
        if in_text.replace(",", "") != from_data.replace(",", ""):
            warn("r3", f"manuscript {in_text!r} vs r3_numbers {from_data!r}")
else:
    warn("r3", "r3_numbers.json not found; run analysis_audit_r3.py")

# ---------------------------------------------------------------------- report
print(f"claim checks: {len(fails)} failures, {len(warns)} to review\n")
for f in fails:
    print("FAIL " + f)
if fails and warns:
    print()
for w in warns:
    print("     " + w)
sys.exit(1 if fails else 0)
