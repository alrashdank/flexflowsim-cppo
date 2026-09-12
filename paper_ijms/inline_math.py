#!/usr/bin/env python3
"""Set every inline symbol in the manuscript as real mathematics.

Before this pass the source carried symbols such as T_min, λ_T and b_{i,t} as
plain text, so the .docx showed literal underscores everywhere outside the
display equations.  This converts them to \( ... \) inline math, which pandoc
turns into OMML in .docx and into TeX in the LaTeX/PDF build, leaving currency
dollars alone (the reason for \( \) rather than $ $).

Skips: the YAML header, display math ($$...$$), inline code (`...`), URLs, and
anything already inside \( \).

    python3 inline_math.py IJMS_manuscript.md          # rewrite in place
    python3 inline_math.py IJMS_manuscript.md --check  # report only
"""
import re
import sys
from pathlib import Path

# Longest / most specific first.  Each pattern is applied to prose segments only.
RULES = [
    # composite phrases -------------------------------------------------------
    (r"P\(TP\(τ\) ≥ T_min\) ≥ 0\.8", r"\(P(\\mathrm{TP}(\\tau) \\ge T_{\\min}) \\ge 0.8\)"),
    (r"P\(min_i u_i\(τ\) ≥ U_min\) ≥ 0\.8", r"\(P(\\min_i u_i(\\tau) \\ge U_{\\min}) \\ge 0.8\)"),
    (r"max\(0, T_min − TP\(τ\)\)", r"\(\\max(0,\\, T_{\\min} - \\mathrm{TP}(\\tau))\)"),
    (r"\(T_min − TP\(τ\)\)/H", r"\((T_{\\min} - \\mathrm{TP}(\\tau))/H\)"),
    (r"−κ C\(τ\)/C_norm", r"\(-\\kappa\\, C(\\tau)/C_{\\mathrm{norm}}\)"),
    (r"C\(τ\) = Σ_t c_t", r"\(C(\\tau) = \\sum_t c_t\)"),
    (r"C\(τ\)/TP\(τ\)", r"\(C(\\tau)/\\mathrm{TP}(\\tau)\)"),
    (r"a = \(a₁, …, a_N\)", r"\(a = (a_1, \\ldots, a_N)\)"),
    (r"∏_s m_s", r"\(\\prod_s m_s\)"),
    (r"\(C_norm, N_norm, W_norm\) = \(2740, 20, 3220\)", r"\((C_{\\mathrm{norm}}, N_{\\mathrm{norm}}, W_{\\mathrm{norm}}) = (2740, 20, 3220)\)"),
    (r"\(C_norm, N_norm, W_norm\)", r"\((C_{\\mathrm{norm}}, N_{\\mathrm{norm}}, W_{\\mathrm{norm}})\)"),
    (r"\(w_c, w_t, w_w\)", r"\((w_c, w_t, w_w)\)"),
    (r"\(λ_T, λ_U\) = \(200, 5\)", r"\((\\lambda_T, \\lambda_U) = (200, 5)\)"),
    (r"r̃_t = r_t − p_t", r"\(\\tilde r_t = r_t - p_t\)"),
    (r"d_t / t", r"\(d_t/t\)"),
    (r"TP − T_min", r"\(\\mathrm{TP} - T_{\\min}\)"),
    (r"TP ≥ T_min", r"\(\\mathrm{TP} \\ge T_{\\min}\)"),
    (r"TP ≥ 18", r"\(\\mathrm{TP} \\ge 18\)"),
    (r"TP = 50", r"\(\\mathrm{TP} = 50\)"),
    (r"η_T = 4000", r"\(\\eta_T = 4000\)"),
    (r"T_min = 18", r"\(T_{\\min} = 18\)"),
    (r"T_min = 50", r"\(T_{\\min} = 50\)"),
    (r"T_min = (\d+)", r"\(T_{\\min} = \1\)"),
    (r"U_min = 0\.(\d+)", r"\(U_{\\min} = 0.\1\)"),
    (r"H = 480", r"\(H = 480\)"),
    (r"κ = 10", r"\(\\kappa = 10\)"),
    (r"Δt = 1 min", r"\(\\Delta t = 1\) min"),
    (r"γ = 0\.99", r"\(\\gamma = 0.99\)"),
    (r"α = 0\.05", r"\(\\alpha = 0.05\)"),
    (r"σ = 0\.1", r"\(\\sigma = 0.1\)"),
    (r"t ≈ 480", r"\(t \\approx 480\)"),
    (r"b_\{i,t\} ∈ \{0, 1\}", r"\(b_{i,t} \\in \\{0, 1\\}\)"),
    # single symbols -----------------------------------------------------------
    (r"(?<![\w\\])T_min(?!\w)", r"\(T_{\\min}\)"),
    (r"(?<![\w\\])U_min(?!\w)", r"\(U_{\\min}\)"),
    (r"λ_\{T,max\}", r"\(\\lambda_{T,\\max}\)"),
    (r"λ_\{U,max\}", r"\(\\lambda_{U,\\max}\)"),
    (r"λ_max", r"\(\\lambda_{\\max}\)"),
    (r"λ_T", r"\(\\lambda_T\)"),
    (r"λ_U", r"\(\\lambda_U\)"),
    (r"η_T", r"\(\\eta_T\)"),
    (r"η_U", r"\(\\eta_U\)"),
    (r"F_fast", r"\(F_{\\mathrm{fast}}\)"),
    (r"C_norm", r"\(C_{\\mathrm{norm}}\)"),
    (r"N_norm", r"\(N_{\\mathrm{norm}}\)"),
    (r"W_norm", r"\(W_{\\mathrm{norm}}\)"),
    (r"g_\{?T,t\}?", r"\(g_{T,t}\)"),
    (r"g_\{?U,t\}?", r"\(g_{U,t}\)"),
    (r"b_\{i,t\}", r"\(b_{i,t}\)"),
    (r"ū_i\(τ\)", r"\(\\bar u_i(\\tau)\)"),
    (r"u_i\(τ\)", r"\(u_i(\\tau)\)"),
    (r"(?<![\w\\])r̃_t(?!\w)", r"\(\\tilde r_t\)"),
    (r"(?<![\w\\])ṅ_t(?!\w)", r"\(\\dot n_t\)"),
    (r"(?<![\w\\])δ_t(?!\w)", r"\(\\delta_t\)"),
    (r"(?<![\w\\])([rcpdW])_t(?!\w)", r"\(\1_t\)"),
    (r"(?<![\w\\])m_s(?!\w)", r"\(m_s\)"),
    (r"(?<![\w\\])w_c(?!\w)", r"\(w_c\)"),
    (r"(?<![\w\\])TP\(τ\)", r"\(\\mathrm{TP}(\\tau)\)"),
    (r"(?<![\w\\(])C\(τ\)", r"\(C(\\tau)\)"),
]

PROTECT = re.compile(
    r"(\$\$[\s\S]*?\$\$"          # display math
    r"|`[^`\n]*`"                 # inline code
    r"|https?://\S+"              # URLs
    r"|\\\([\s\S]*?\\\)"          # already-converted inline math
    r"|\[@[^\]]+\])",             # citation keys
    re.S)


def convert(text: str):
    head, sep, body = text.partition("\n---\n")          # keep YAML header intact
    if not sep:
        head, body = "", text
    else:
        head = head + sep
    out, pos, n = [], 0, 0
    for m in PROTECT.finditer(body):
        seg = body[pos:m.start()]
        seg, k = _apply(seg)
        n += k
        out.append(seg)
        out.append(m.group(0))
        pos = m.end()
    seg, k = _apply(body[pos:])
    n += k
    out.append(seg)
    return head + "".join(out), n


def _apply(seg: str):
    total = 0
    for pat, rep in RULES:
        seg, k = re.subn(pat, rep, seg)
        total += k
    return seg, total


if __name__ == "__main__":
    path = Path(sys.argv[1])
    src = path.read_text(encoding="utf-8")
    new, n = convert(src)
    if "--check" in sys.argv:
        print(f"{n} conversions would be made")
    else:
        path.write_text(new, encoding="utf-8")
        print(f"{n} symbols set as inline math in {path.name}")
    # anything left?
    left = re.findall(r"(?<![\\\w])[A-Za-zλητκ]_[A-Za-z{][^\s,.;:)]*", re.sub(PROTECT, "", new))
    left = [x for x in left if not x.startswith(("run_", "lambda_", "protocol_", "analysis_", "results_", "r3_", "fig_", "make_", "pilot_", "verify_", "configs", "n_steps", "ent_", "gae_"))]
    if left:
        print("possible leftovers:", sorted(set(left))[:30])
