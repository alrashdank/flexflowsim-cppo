"""R1_manuscript.md -> TMLR LaTeX (anonymised submission). Body via pandoc; citations to natbib;
appendices after the bibliography; declarations reduced to what TMLR expects."""
import re, subprocess, pathlib
SRC=pathlib.Path("/home/claude/draft/R1_manuscript.md"); s=SRC.read_text()
s=re.sub(r"^---\ntitle:.*?\n---\n","",s,flags=re.S)                      # YAML title
title=re.search(r'title: "(.*?)"',SRC.read_text()).group(1)
# ---- split: abstract | body | back matter | appendices
abstract=s.split("## Abstract\n\n")[1].split("\n\n**Keywords")[0]
keywords=re.search(r"\*\*Keywords:\*\*\s*(.*?)\n\n",s,re.S).group(1).replace("\n"," ")
body=s.split("## 1. Introduction")[1].split("## Data availability")[0]
body="## 1. Introduction"+body        # includes the unnumbered Broader impact statement ({-})
data_avail=s.split("## Data availability")[1].split("## Funding")[0]
genai=s.split("## Declaration of generative AI use")[1].split("## CRediT")[0]
appx=s.split("## Appendix A. Reproducibility")[1]
appx="## Appendix A. Reproducibility"+appx
body=body.replace("r̃_t","$\\tilde r_t$")
# ---- inline mathematics in running text (outside $...$ and code spans)
PHRASES=[("P(TP(τ) ≥ T_min) ≥ 0.8 and P(min_i u_i(τ) ≥ U_min) ≥ 0.8","$P(\\mathrm{TP}(\\tau) \\ge T_{\\min}) \\ge 0.8$ and $P(\\min_i u_i(\\tau) \\ge U_{\\min}) \\ge 0.8$"),
("E[C(τ)/TP(τ)]","$\\mathbb{E}[C(\\tau)/\\mathrm{TP}(\\tau)]$"),
("g_{U,t} = Σ_{i∈F_fast} max(0, U_min − ū_{i,t})","$g_{U,t} = \\sum_{i\\in F_{\\mathrm{fast}}} \\max(0,\\, U_{\\min} - \\bar u_{i,t})$"),
("C(τ) = Σ_t c_t","$C(\\tau) = \\sum_t c_t$"),
("−κ C(τ)/C_norm","$-\\kappa\\, C(\\tau)/C_{\\mathrm{norm}}$"),
("|Σ p_t| / |Σ r_t|","$|\\sum_t p_t| / |\\sum_t r_t|$"),
("Σp_t","$\\sum_t p_t$"),
("(λ_T, λ_U) = (200, 5)","$(\\lambda_T, \\lambda_U) = (200, 5)$"),
("max(0, T_min − TP(τ))","$\\max(0,\\, T_{\\min} - \\mathrm{TP}(\\tau))$"),
("(T_min − TP(τ))/H","$(T_{\\min} - \\mathrm{TP}(\\tau))/H$"),
("TP − T_min","$\\mathrm{TP} - T_{\\min}$"),
("TP(τ) ≥ T_min","$\\mathrm{TP}(\\tau) \\ge T_{\\min}$"),
("TP ≥ 18","$\\mathrm{TP} \\ge 18$"),("TP ≥ T_min","$\\mathrm{TP} \\ge T_{\\min}$"),
("a = (a₁, …, a_N)","$a = (a_1, \\ldots, a_N)$"),("∏_s m_s","$\\prod_s m_s$"),("m = (2, 2)","$m = (2, 2)$"),("m = (2, 3, 2)","$m = (2, 3, 2)$"),
("(w_c, w_t, w_w)","$(w_c, w_t, w_w)$"),("(C_norm, N_norm, W_norm)","$(C_{\\mathrm{norm}}, N_{\\mathrm{norm}}, W_{\\mathrm{norm}})$"),
("d_t / t","$d_t/t$"),("ū_{i,t}","$\\bar u_{i,t}$"),("ū_i(τ)","$\\bar u_i(\\tau)$"),("u_i(τ)","$u_i(\\tau)$"),("TP(τ)","$\\mathrm{TP}(\\tau)$"),("C(τ)","$C(\\tau)$"),
("b_{i,t} ∈ {0, 1}","$b_{i,t} \\in \\{0, 1\\}$"),("b_{i,t}","$b_{i,t}$"),("g_{U,t}","$g_{U,t}$"),("g_T,t","$g_{T,t}$"),("g_U,t","$g_{U,t}$"),
("λ_{T,max}","$\\lambda_{T,\\max}$"),("λ_{U,max}","$\\lambda_{U,\\max}$"),
("λ_T","$\\lambda_T$"),("λ_U","$\\lambda_U$"),("η_T","$\\eta_T$"),("η_U","$\\eta_U$"),("T_min","$T_{\\min}$"),("U_min","$U_{\\min}$"),
("F_fast","$F_{\\mathrm{fast}}$"),("C_norm","$C_{\\mathrm{norm}}$"),("N_norm","$N_{\\mathrm{norm}}$"),("W_norm","$W_{\\mathrm{norm}}$"),
("ṅ_t","$\\dot n_t$"),("δ_t","$\\delta_t$"),("m_s","$m_s$"),("d_t","$d_t$"),("c_t","$c_t$"),("r_t","$r_t$"),("p_t","$p_t$"),("W_t","$W_t$"),
("w_c","$w_c$"),("w_t","$w_t$"),("w_w","$w_w$"),("a_N","$a_N$"),("min_i","$\\min_i$"),("g_T ","$g_T$ "),("Δt","$\\Delta t$")]
def mathify(md):
    for a,b in PHRASES:                      # re-split after every phrase so inserted math is never re-edited
        parts=re.split(r"(\$\$.*?\$\$|\$[^$\n]+\$|`[^`\n]+`)",md,flags=re.S)
        md="".join(p if i%2==1 else p.replace(a,b) for i,p in enumerate(parts))
    return md
body=mathify(body); abstract=mathify(abstract); appx=mathify(appx)
# ---- citations -> natbib
CIT={"Achiam et al., 2017":"achiam2017cpo","Agarwal et al., 2021":"agarwal2021precipice","Ali & Tirel, 2023":"ali2023masked",
"Altman, 1999":"altman1999cmdp","Author, 2026a":"author2026a","Author, 2026b":"author2026b","Babor & Hitzmann, 2022":"babor2022bakery",
"Doherty et al., 2025":"doherty2025hype","Ferreira et al., 2022":"ferreira2022dispatching","Henderson et al., 2018":"henderson2018matters",
"Huang et al., 2025":"huang2025evolving","Li et al., 2025":"li2025evolutionary","Liu et al., 2025":"liu2025gat","Marques et al., 2025":"marques2025dynamic",
"Mayerhoff & Schmidt, 2026":"mayerhoff2026slr","Paternain et al., 2019":"paternain2019duality","Raffin et al., 2021":"raffin2021sb3",
"Rinciog & Meyer, 2021":"rinciog2021fabricatio","Schneider et al., 2026":"schneider2026role","Schulman et al., 2017":"schulman2017ppo",
"Shen et al., 2026":"shen2026transformer","Stooke et al., 2020":"stooke2020pid","Tang et al., 2020":"tang2020mask","Tessler et al., 2019":"tessler2019rcpo",
"Wang et al., 2025":"wang2025end","Zhang et al., 2025":"zhang2025dual"}
def norm(t): return re.sub(r"\s+"," ",t)
def cite_paren(m):
    inner=norm(m.group(1))
    pre=""
    if ";" in inner and not re.match(r"^[A-Z][\w\-]+(?: (?:&|and) [A-Z][\w\-]+| et al\.)?, (?:19|20)\d\d[ab]?",inner.split(";")[0].strip()):
        pre,inner=inner.split(";",1); pre=pre.strip(); inner=inner.strip()      # "(PPO; Schulman et al., 2017)"
    post=""
    parts=[p.strip() for p in inner.split(";")]
    keys=[]
    for p in parts:
        if p in CIT: keys.append(CIT[p])
        elif parts.index(p)==len(parts)-1 and keys: post=p                     # trailing note
        else: return m.group(0)                                                 # not a citation
    if not keys: return m.group(0)
    if pre: return f"({pre}; \\citealp{{{','.join(keys)}}})"
    if post: return f"\\citep[][{post}]{{{','.join(keys)}}}"
    return f"\\citep{{{','.join(keys)}}}"
body=re.sub(r"\(([^()]*?\b(?:19|20)\d\d[ab]?[^()]*?)\)",cite_paren,body)
def cite_text(m):
    name=norm(m.group(1)).replace(" and ","& ").replace("& ","& "); key=CIT.get(f"{name}, {m.group(2)}".replace("& ","& "))
    if key is None:
        key=CIT.get(f"{norm(m.group(1)).replace(' and ',' & ')}, {m.group(2)}")
    return f"\\citet{{{key}}}" if key else m.group(0)
body=re.sub(r"\b([A-Z][\w\-]+(?:\s(?:and|&)\s[A-Z][\w\-]+|\set al\.)?)\s\(((?:19|20)\d\d[ab]?)\)",cite_text,body)
for _n in ("data_avail","appx"):
    _v=globals()[_n]; _v=re.sub(r"\(([^()]*?\b(?:19|20)\d\d[ab]?[^()]*?)\)",cite_paren,_v)
    _v=re.sub(r"\b([A-Z][\w\-]+(?:\s(?:and|&)\s[A-Z][\w\-]+|\set al\.)?)\s\(((?:19|20)\d\d[ab]?)\)",cite_text,_v); globals()[_n]=_v
assert "\\citet{rinciog2021fabricatio}" in body, "Rinciog"
leftover=re.findall(r"\b(?:Altman|Schulman|Doherty|Agarwal|Henderson|Paternain|Stooke|Author) \(?(?:19|20)\d\d",body)
print("unconverted narrative cites:",leftover)
# ---- headings: strip manual numbering (LaTeX numbers sections); keep appendix letters via \appendix
body=re.sub(r"^## \d+\. ","## ",body,flags=re.M); body=re.sub(r"^### \d+\.\d+ ","### ",body,flags=re.M)
appx=re.sub(r"^## Appendix [AB]\. ","## ",appx,flags=re.M); appx=re.sub(r"^### [AB]\.\d ","### ",appx,flags=re.M)
# ---- figures: pandoc handles ![](x.png){width=16cm}; switch to pdf + textwidth
for txt in ("body",):
    pass
body=body.replace("| electronics, symmetric cell (§6.7) |","| electronics, symmetric |")
body=body.replace("![](fig_lambda_T.png){width=16cm}\\","![](fig_lambda_T.pdf){width=100%}").replace("![](fig_symmetric.png){width=16cm}\\","![](fig_symmetric.pdf){width=100%}")
def proportional_tables(md):
    out=[]; lines=md.split("\n"); i=0
    while i<len(lines):
        if lines[i].startswith("|") and i+1<len(lines) and re.match(r"^\|[-: |]+\|$",lines[i+1]):
            j=i
            while j<len(lines) and lines[j].startswith("|"): j+=1
            rows=[l for l in lines[i:j] if not re.match(r"^\|[-: |]+\|$",l)]
            cells=[[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            ncol=max(len(c) for c in cells)
            w=[max(len(c[k]) if k<len(c) else 0 for c in cells) for k in range(ncol)]
            w=[max(3,min(x,26)) for x in w]                       # cap very long cells; they wrap
            sep="|"+"|".join("-"*x for x in w)+"|"
            out.extend([lines[i],sep]+lines[i+2:j]); i=j
        else:
            out.append(lines[i]); i+=1
    return "\n".join(out)

def md2tex(md):
    md=proportional_tables(md)
    r=subprocess.run(["pandoc","-f","markdown","-t","latex","--wrap=none","--shift-heading-level-by=-1","--columns=60"],input=md,text=True,capture_output=True); assert r.returncode==0,r.stderr; return r.stdout
tex_body=md2tex(body); tex_appx=md2tex(appx); tex_abs=md2tex(abstract).strip(); tex_data=md2tex(data_avail).strip(); tex_genai=md2tex(genai).strip()
for t in ("tex_body","tex_appx"):
    v=globals()[t]
    v=v.replace("\\begin{longtable}[]{@{}","{\\footnotesize\\setlength{\\tabcolsep}{3pt}\\begin{longtable}[]{@{}").replace("\\end{longtable}","\\end{longtable}}")   # small type for tables
    v=re.sub(r"\\includegraphics\[width=1\\textwidth,height=\\textheight\]","\\\\includegraphics[width=\\\\textwidth]",v)
    globals()[t]=v
def floats(v):
    # tables: bold caption paragraph + pandoc longtable -> table float with caption above
    pat=re.compile(r"\\textbf\{Table (\d+)\.\} (.*?)\n\n\{\\footnotesize\\setlength\{\\tabcolsep\}\{3pt\}\\begin\{longtable\}\[\]\{(.*?)\}\n(.*?)\\endhead\n\\bottomrule\\noalign\{\}\n\\endlastfoot\n(.*?)\\end\{longtable\}\}",re.S)
    def rt(m):
        n,cap,spec,head,body=m.groups()
        head=head.replace("\\midrule\\noalign{}\n","\\midrule\n").replace("\\toprule\\noalign{}\n","\\toprule\n")
        return ("\\begin{table}[htbp]\n\\caption{"+cap.strip()+"}\\label{tab:"+n+"}\n\\vspace{3pt}\n{\\footnotesize\\setlength{\\tabcolsep}{3pt}\\centering\n\\begin{tabular}{"+spec+"}\n"+head+body+"\\bottomrule\n\\end{tabular}}\n\\end{table}")
    v,nt=pat.subn(rt,v)
    # figures: includegraphics + bold caption paragraph -> figure float with caption below
    pf=re.compile(r"(\\includegraphics\[[^\]]*\]\{[^}]*\})\n\n\\textbf\{Figure (\d+)\.\} (.*?)\n",re.S)
    v,nf=pf.subn(lambda m:"\\begin{figure}[htbp]\n\\centering\n"+m.group(1)+"\n\\caption{"+m.group(3).strip()+"}\\label{fig:"+m.group(2)+"}\n\\end{figure}\n",v)
    print("floats:",nt,"tables",nf,"figures"); return v
tex_body=floats(tex_body)
# equations: pandoc emits \[...\qquad(1)\]; fine in TMLR.
title_tex=title.replace(': ', ':\\\\ ',1)
doc=f"""\\documentclass[10pt]{{article}}
\\usepackage{{tmlr}}
\\usepackage[hidelinks]{{hyperref}}
\\usepackage{{url}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{array}}
\\usepackage{{calc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{textcomp}}
\\usepackage{{newunicodechar}}
\\newunicodechar{{λ}}{{\\ensuremath{{\\lambda}}}}\\newunicodechar{{η}}{{\\ensuremath{{\\eta}}}}\\newunicodechar{{κ}}{{\\ensuremath{{\\kappa}}}}\\newunicodechar{{γ}}{{\\ensuremath{{\\gamma}}}}\\newunicodechar{{τ}}{{\\ensuremath{{\\tau}}}}\\newunicodechar{{π}}{{\\ensuremath{{\\pi}}}}\\newunicodechar{{Σ}}{{\\ensuremath{{\\Sigma}}}}\\newunicodechar{{Δ}}{{\\ensuremath{{\\Delta}}}}\\newunicodechar{{δ}}{{\\ensuremath{{\\delta}}}}\\newunicodechar{{σ}}{{\\ensuremath{{\\sigma}}}}\\newunicodechar{{α}}{{\\ensuremath{{\\alpha}}}}\\newunicodechar{{ū}}{{\\ensuremath{{\\bar{{u}}}}}}\\newunicodechar{{ṅ}}{{\\ensuremath{{\\dot{{n}}}}}}
\\newunicodechar{{≥}}{{\\ensuremath{{\\geq}}}}\\newunicodechar{{≤}}{{\\ensuremath{{\\leq}}}}\\newunicodechar{{×}}{{\\ensuremath{{\\times}}}}\\newunicodechar{{−}}{{\\ensuremath{{-}}}}\\newunicodechar{{±}}{{\\ensuremath{{\\pm}}}}\\newunicodechar{{→}}{{\\ensuremath{{\\rightarrow}}}}\\newunicodechar{{∈}}{{\\ensuremath{{\\in}}}}\\newunicodechar{{≈}}{{\\ensuremath{{\\approx}}}}\\newunicodechar{{⁻}}{{\\ensuremath{{^{{-}}}}}}\\newunicodechar{{⁴}}{{\\ensuremath{{^{{4}}}}}}\\newunicodechar{{³}}{{\\ensuremath{{^{{3}}}}}}\\newunicodechar{{⁰}}{{\\ensuremath{{^{{0}}}}}}\\newunicodechar{{¹}}{{\\ensuremath{{^{{1}}}}}}\\newunicodechar{{²}}{{\\ensuremath{{^{{2}}}}}}\\newunicodechar{{₁}}{{\\ensuremath{{_{{1}}}}}}\\newunicodechar{{§}}{{\\S}}\\newunicodechar{{∀}}{{\\ensuremath{{\\forall}}}}\\newunicodechar{{∏}}{{\\ensuremath{{\\prod}}}}
\\providecommand{{\\tightlist}}{{\\setlength{{\\itemsep}}{{0pt}}\\setlength{{\\parskip}}{{0pt}}}}
\\setlength{{\\LTcapwidth}}{{\\textwidth}}
\\title{{{title_tex}}}
\\author{{\\name Anonymous authors \\email \\\\ \\addr Paper under double-blind review}}
\\def\\month{{MM}}\\def\\year{{YYYY}}\\def\\openreview{{\\url{{https://openreview.net/forum?id=XXXX}}}}
\\begin{{document}}
\\maketitle
\\begin{{abstract}}
{tex_abs}
\\end{{abstract}}
{tex_body}
\\section*{{Data and code availability}}
{tex_data}
\\section*{{Use of large language models}}
{tex_genai}
\\bibliographystyle{{tmlr}}
\\bibliography{{refs}}
\\appendix
{tex_appx}
\\end{{document}}
"""
pathlib.Path("main.tex").write_text(doc); print("written main.tex", len(doc))
