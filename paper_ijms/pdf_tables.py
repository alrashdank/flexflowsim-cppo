"""Abbreviate long table headers in the reading-PDF source so that pandoc's
LaTeX tables do not run header cells together.  Mirrors the HDR map used for
the docx in postprocess_docx.py.  Usage: python3 pdf_tables.py in.md out.md"""
import re
import sys

HDR = {
    "Selected checkpoint": "Ckpt", "Stochastic CPU ± CI": "Stoch. CPU ± CI",
    "Stochastic joint sat.": "Stoch. joint sat.", "Stochastic CPU": "Stoch. CPU",
    "Stochastic TP": "Stoch. TP", "Greedy CPU ± CI": "Greedy CPU ± CI",
    "Greedy joint sat.": "Greedy sat.", "Q1 greedy CPU ± CI": "Q1 greedy CPU ± CI",
    "Q1 greedy joint sat.": "Q1 greedy sat.", "Q1 stochastic CPU ± CI": "Q1 stoch. CPU ± CI",
    "Q1 stochastic joint sat.": "Q1 stoch. sat.", "Q2 stochastic CPU ± CI": "Q2 stoch. CPU ± CI",
    "Q2 joint sat.": "Q2 sat.", "Q2 val.-satisfied": "Q2 val.-sat.", "Val.-satisfied": "Val.-sat.",
    "Penalty ratio": "Pen. ratio", "Stage capacity": "Capacity",
}
lines = open(sys.argv[1], encoding="utf-8").read().split("\n")
for i in range(len(lines) - 1):
    if lines[i].startswith("|") and re.match(r"^\|[\s:|-]+\|$", lines[i + 1]):
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        cells = [HDR.get(c, c) for c in cells]
        lines[i] = "| " + " | ".join(cells) + " |"
open(sys.argv[2], "w", encoding="utf-8").write("\n".join(lines))
