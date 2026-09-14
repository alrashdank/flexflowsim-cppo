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

# ----------------------------------------------- 6 cross-references must resolve
# Every "Section x.y", "Table N", "Figure N", "Eq. (n)" and "Appendix X" cited in
# the text must point at something that exists, and every table and figure must
# be cited at least once outside its own caption.
HEADINGS = set()
for l in LINES:
    m = re.match(r"#{2,3}\s+(\d+(?:\.\d+)?)\.?\s", l)
    if m:
        HEADINGS.add(m.group(1))
    m = re.match(r"\*\*([A-C]\.\d+)\s", l)          # appendix sub-sections **A.2 ...**
    if m:
        HEADINGS.add(m.group(1))
    m = re.match(r"##\s+Appendix\s+([A-C])\b", l)
    if m:
        HEADINGS.add(m.group(1))
for i, l in locate(r"Sections?\s+\d"):
    m = re.search(r"Sections?\s+((?:\d+(?:\.\d+)?)(?:(?:,\s*|\s+and\s+)\d+(?:\.\d+)?(?!\s*%))*)", l)
    for ref in re.findall(r"\d+(?:\.\d+)?", m.group(1)):
        if ref not in HEADINGS and not (ref.isdigit() and 1 <= int(ref) <= 9):
            fail("xref", f"L{i}: Section {ref} does not exist -> {l[:90]}")
for i, l in locate(r"Appendix\s+[A-C]\b"):
    for ref in re.findall(r"Appendix\s+([A-C](?:\.\d+)?)", l):
        if ref not in HEADINGS:
            fail("xref", f"L{i}: Appendix {ref} does not exist -> {l[:90]}")
for i, l in locate(r"\b(?:Section|Sections)\s+([A-C]\.\d+)"):
    for ref in re.findall(r"\b(?:Section|Sections)\s+([A-C]\.\d+)", l):
        if ref not in HEADINGS:
            fail("xref", f"L{i}: Section {ref} does not exist -> {l[:90]}")

CAPTIONS = {m.group(1) for m in re.finditer(r"^\*\*(Table [A-C]?\d+|Figure \d+)\.\*\*", TEXT, re.M)}
CITED = {}
for i, l in enumerate(LINES, 1):
    if re.match(r"^\*\*(Table [A-C]?\d+|Figure \d+)\.\*\*", l):
        continue                                  # the caption itself
    for m in re.finditer(r"\b(Tables?|Figures?)\s+([A-C]?\d+(?:\s*(?:,|and)\s*[A-C]?\d+)*)", l):
        kind = "Table" if m.group(1).startswith("Table") else "Figure"
        tail = l[m.end(): m.end() + 1]
        nums = re.findall(r"[A-C]?\d+", m.group(2))
        if tail == "%":                              # "Table 1 and 95%": drop the percentage
            nums = nums[:-1]
        for n in nums:
            CITED.setdefault(f"{kind} {n}", []).append(i)
for c in sorted(CITED):
    if c not in CAPTIONS:
        fail("xref", f"{c} is cited (L{CITED[c][:3]}) but has no caption")
for c in sorted(CAPTIONS):
    if c not in CITED:
        fail("xref", f"{c} has a caption but is never cited in the text")

EQ_TAGS = {m.group(1) for m in re.finditer(r"\\qquad\((\d+)\)\$\$", TEXT)}
for i, l in locate(r"Eqs?\.\s*\("):
    for n in re.findall(r"\((\d+)\)", l):
        if n not in EQ_TAGS and re.search(rf"Eqs?\.\s*\(\s*{n}\s*\)|and\s*\({n}\)|\({n}\)[–-]", l):
            fail("xref", f"L{i}: Eq. ({n}) is cited but no display equation carries that tag")

# ----------------------------------------------------- 7 wording that must be gone
# Each of these was a defect found by one of the three audits; none may return.
BANNED = [
    (r"\boracle\b", "ShortestQueue is a fixed dispatching rule, not an oracle"),
    (r"\btoward\b(?!s)", "British spelling is 'towards'"),
    (r"Hitzmann", "the dataset has a single creator (DataCite)"),
    (r"draft\s*[—-]+\s*please check", "draft note must not ship"),
    (r"without approximation|telescopes exactly", "the optimised return is discounted (gamma = 0.99)"),
    (r"pre-specified family of ten|prespecified family of ten", "the family of ten was defined after the R1 runs"),
    (r"\bpublished (?:signal|cell|wrapper|configuration)\b", "the initial study is unpublished"),
    (r"\b(?:was|were|has been|have been) (?:published|submitted)\b", "the initial study is unpublished; check the sentence"),
    (r"infeasible for any policy", "a 0.50 floor on the third-stage fast server is feasible (about 0.6 if all work is routed to it)"),
    (r"truncated[- ]normal|truncated below", "service times are clipped normal draws (env.py: max(normal, floor))"),
    (r"is the joint satisfaction rate itself", "negative penalty is implied by joint satisfaction, not equal to it"),
    (r"does not\s+cross until", "crossing time is not determined by the final throughput"),
    (r"explains the residual result", "narrow to 'accounts for the direction of'"),
    (r"on the stated CMDP therefore", "the cell optimises a discounted, aggregate-utilisation surrogate"),
    (r"for practical purposes it\s+was that objective", "say 'numerically'"),
    (r"\b(?:both|either) metrics?\b", "name the two metrics"),
    (r"in this literature almost always", "unsupported generalisation; say 'here'"),
    (r"so no comparison is affected", "identical procedure does not establish no differential effect"),
    (r"no longer drives the multiplier to its cap", "limit to the tested budget"),
    (r"Four are state-aware", "CostMinimising and FastServerFirst are fixed-route rules"),
    (r"lies? between (?:this testbed|the two)", "0.06-0.13 is not between 0.03 and 0.1"),
    (r"on a capacity-limited line total cost is nearly flat", "keep the generalisation proportional: 'these lines'"),
    (r"implies a negative penalty", "equality gives zero: 'non-positive'"),
    (r"must be expressed per\s+step", "terminal rewards are possible: 'this implementation expresses per step'"),
]
for pat, why in BANNED:
    for i, l in locate(pat):
        fail("banned", f"L{i}: /{pat}/ ({why}) -> {l[:90]}")
for i, l in locate(r"pre-specified"):
    ctx = " ".join(LINES[max(0, i - 2): i + 2]).lower()
    if re.search(r"(?:outside any|no|not|without a)\s+pre-specified", ctx, re.I):
        continue                                   # says there was none: correct
    if "twelve" not in ctx and "protocol" not in ctx:
        fail("banned", f"L{i}: 'pre-specified' without the family of twelve / protocol in context -> {l[:90]}")

# --------------------------------------------- 8 the built files must match the text
DRAFT = Path(sys.argv[1]).resolve().parent
docx_path = DRAFT / "IJMS_manuscript.docx"
if docx_path.exists():
    try:
        from docx import Document
        from docx.oxml.ns import qn
        d = Document(str(docx_path))
        body = d.element.body
        n_omath = len(body.findall(".//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath"))
        n_src_math = TEXT.count("\\(") + TEXT.count("$$") // 2
        if n_omath < 0.9 * n_src_math:
            fail("docx", f"only {n_omath} OMML equations in docx against {n_src_math} math spans in source")
        prose = "\n".join(p.text for p in d.paragraphs)
        for pat in (r"\bT_min\b", r"\bU_min\b", r"λ_T|lambda_T", r"\\\(", r"\\\)", r"\[@", r"\bg_T\b", r"\\mathrm", r"\\min\b"):
            for m in re.finditer(pat, prose):
                fail("docx", f"literal {m.group(0)!r} in docx prose near: {prose[max(0, m.start()-40): m.end()+40]!r}")
                break
        sectPr = d.sections[0]._sectPr
        if sectPr.find(qn("w:lnNumType")) is None:
            fail("docx", "no continuous line numbering in section properties")
        if "PAGE" not in d.sections[0].footer._element.xml:
            fail("docx", "no PAGE field in footer")
        if "draft" in prose.lower() and re.search(r"draft\s*[—-]", prose, re.I):
            fail("docx", "a draft note survives in the docx")
        n_tables_docx = len(d.tables)
        n_tables_src = len([c for c in CAPTIONS if c.startswith("Table")])
        if n_tables_docx != n_tables_src:
            fail("docx", f"{n_tables_docx} tables in docx, {n_tables_src} captions in source")
    except ImportError:
        warn("docx", "python-docx not installed; docx checks skipped")
else:
    warn("docx", f"{docx_path.name} not built; docx checks skipped")

anon_path = DRAFT / "IJMS_manuscript_anonymous.docx"
if anon_path.exists():
    try:
        from docx import Document
        atxt = "\n".join(p.text for p in Document(str(anon_path)).paragraphs).lower()
        for ident in ("khaled", "alrashdank", "paaet", "kuwait", "orcid", "github.com", "kr.alrashdan"):
            if ident in atxt:
                fail("anon", f"identifier {ident!r} present in the anonymous manuscript")
        if "the author's own" in atxt:
            fail("anon", "'the author's own' present in the anonymous manuscript")
    except ImportError:
        pass

pdf_path = DRAFT / "IJMS_manuscript.pdf"
if pdf_path.exists():
    import subprocess
    txt = subprocess.run(["pdftotext", "-layout", str(pdf_path), "-"],
                         capture_output=True, text=True).stdout
    for n in sorted(EQ_TAGS, key=int):
        if f"({n})" not in txt:
            fail("pdf", f"equation tag ({n}) missing from PDF text")
    for bad in ("[@", "\\(", "\\)", "??", "\\mathrm", "\\lambda"):
        if bad in txt:
            fail("pdf", f"literal {bad!r} in PDF text")
    if "λ" not in txt and "\U0001d706" not in txt:      # upright or mathematical-italic lambda
        fail("pdf", "no λ glyph in PDF text; equations may not have rendered")
    for c in sorted(CAPTIONS):
        if not re.search(rf"{c}\.", txt):
            fail("pdf", f"caption '{c}.' missing from PDF")
    for tif in ("Figure1.tif", "Figure2.tif"):
        if not (DRAFT / tif).exists():
            fail("pdf", f"{tif} missing")
else:
    warn("pdf", f"{pdf_path.name} not built; PDF checks skipped")

fig_script = Path("/home/claude/repo/make_fig_lambda.py")
if fig_script.exists() and re.search(r'"[^"]*published[^"]*"', fig_script.read_text()):
    fail("figure", "make_fig_lambda.py labels a curve 'published'")

title_page = DRAFT / "IJMS_title_page.md"
if title_page.exists():
    tp = title_page.read_text()
    if re.search(r"draft\s*[—-]+\s*please", tp, re.I):
        fail("title", "draft note survives in the title page")
    m = re.search(r"\*\*Word count:\*\*\s*([\d,]+)\s+words", tp)
    if m and docx_path.exists():
        stated = int(m.group(1).replace(",", ""))
        try:
            from docx import Document
            d2 = Document(str(docx_path))
            words = sum(len(p.text.split()) for p in d2.paragraphs)
            words += sum(len(c.text.split()) for t in d2.tables for r in t.rows for c in r.cells)
            if abs(words - stated) > 0.03 * words:
                fail("title", f"title page states {stated} words; docx holds about {words}")
        except ImportError:
            pass

# ---------------------------------------------- 9 remaining r3 figures in the text
if r3p.exists():
    fb, fe = r3["fill"]["bakery"], r3["fill"]["electronics"]
    more = [
        (f"{fb['crossing_median']:.0f}", "bakery crossing median"),
        (f"{round(fe['crossing_median'] + 1e-9):.0f}", "electronics crossing median"),
        (f"{fb['crossing_min']} to {fb['crossing_max']}", "bakery crossing range"),
        (f"{fe['crossing_min']} to {fe['crossing_max']}", "electronics crossing range"),
    ]
    ab = r3["ablate"]
    def fmt(v, dec):
        return f"{v:+.{dec}f}".replace("-", "−").replace("+0.", "+0.")
    for key, dec in (("cumrate: cost-only minus shaped, cost per unit ($)", 2),
                     ("episode: cost-only minus shaped, cost per unit ($)", 2),
                     ("cumrate: cost-only minus shaped, joint satisfaction (pp)", 1),
                     ("episode: cost-only minus shaped, joint satisfaction (pp)", 1)):
        r = ab[key]
        more.append((f"{fmt(r['difference'], dec)} [{fmt(r['ci_low'], dec)}, {fmt(r['ci_high'], dec)}]", key))
    for in_text, what in more:
        pat = re.escape(in_text).replace(r"\[", r"\$?\[").replace("−", "−\\$?").replace(r"\+", r"\+\$?")
        if in_text not in TEXT and not re.search(pat, TEXT):
            fail("r3", f"{what}: {in_text!r} absent from manuscript")
    if "negpen" in r3:
        rules = ("ShortestQueue", "LeastUtilised", "RoundRobin", "UniformRandom")
        lo = min(min(r3["negpen"][tb][r]["negative_at_final_by_seed"]) for tb in r3["negpen"] for r in rules)
        hi = max(max(r3["negpen"][tb][r]["negative_at_final_by_seed"]) for tb in r3["negpen"] for r in rules)
        clo = min(r3["negpen"][tb]["corrected (own final multipliers)"]["negative"] for tb in r3["negpen"])
        chi = max(r3["negpen"][tb]["corrected (own final multipliers)"]["negative"] for tb in r3["negpen"])
        for in_text, what in ((f"{100*lo:.0f}–{100*hi:.0f}% of the rules' episodes", "negative-penalty range, rules"),
                              (f"{100*clo:.0f}–{100*chi:.0f}%\nof the corrected cell's", "negative-penalty range, corrected")):
            if in_text not in TEXT and in_text.replace("\n", " ") not in TEXT:
                fail("r3", f"{what}: {in_text!r} absent from manuscript")
        fb2 = r3["fill"]["bakery"]
        fc, ba = fb2["at_floor_first_crossing"], fb2["at_floor_steps_below_after"]
        for in_text, what in ((f"steps {min(fc)} to {max(fc)}", "at-floor first-crossing range"),
                              (f"{min(ba)} to {max(ba)}\nfurther steps", "at-floor steps below after crossing"),
                              (f"{fb2['spearman_tp_vs_first_crossing']:.2f}".replace("-", "−"), "Spearman bakery"),
                              (f"{r3['fill']['electronics']['spearman_tp_vs_first_crossing']:.2f}".replace("-", "−"), "Spearman electronics")):
            if in_text not in TEXT and in_text.replace("\n", " ") not in TEXT:
                fail("r3", f"{what}: {in_text!r} absent from manuscript")
    sel = r3["symsat"]["selection"]
    for v, what in ((sel["checkpoints_validated"], "checkpoints validated"),
                    (sel["checkpoints_qualifying"], "checkpoints qualifying")):
        if not re.search(rf"\b{v}\b", TEXT):
            fail("r3", f"{what} = {v} absent from manuscript")

# ---------------------------------------------------------------------- report
print(f"claim checks: {len(fails)} failures, {len(warns)} to review\n")
for f in fails:
    print("FAIL " + f)
if fails and warns:
    print()
for w in warns:
    print("     " + w)
sys.exit(1 if fails else 0)
