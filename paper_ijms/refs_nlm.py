#!/usr/bin/env python3
"""Convert [@key] citations in IJMS_manuscript.md to T&F Standard NLM numbered style,
number in order of first appearance, emit the numbered reference list, and reorder the
document into the running order the journal specifies (tables and figures at the end)."""
import re, sys, pathlib

NLM = {
"achiam2017cpo": "Achiam J, Held D, Tamar A, et al. Constrained policy optimization. In: Proceedings of the 34th International Conference on Machine Learning. PMLR 70; 2017. p. 22–31.",
"agarwal2021precipice": "Agarwal R, Schwarzer M, Castro PS, et al. Deep reinforcement learning at the edge of the statistical precipice. In: Advances in Neural Information Processing Systems 34; 2021. p. 29304–29320.",
"ali2023masked": "Ali AM, Tirel L. Action masked deep reinforcement learning for controlling industrial assembly lines. In: 2023 IEEE World AI IoT Congress (AIIoT); 2023. p. 797–803. doi: 10.1109/AIIoT58121.2023.10174426",
"alrashdan2026breakdowns": "Alrashdan KR. Routing under machine breakdowns: a benchmark of dispatching rules, bandits, and reinforcement learning for multi-server flow shops. J King Saud Univ Eng Sci. 2026;38(7):55. doi: 10.1007/s44444-026-00128-9",
"altman1999cmdp": "Altman E. Constrained Markov decision processes. Boca Raton (FL): Chapman & Hall/CRC; 1999.",
"babor2022bakery": "Babor M. Small and medium-sized bakery production data for scheduling. Version 2 [dataset]. 2022 [cited 2026 Sep 12]. In: Mendeley Data [Internet]. Available from: https://doi.org/10.17632/dhgbssb8ns.2",
"doherty2025hype": "Doherty M, Matzner R, Sadeghi R, et al. Reinforcement learning for dynamic resource allocation in optical networks: hype or hope? J Opt Commun Netw. 2025;17(9):D1–D17. doi: 10.1364/JOCN.559990",
"ferreira2022dispatching": "Ferreira C, Figueira G, Amorim P. Effective and interpretable dispatching rules for dynamic job shops via guided empirical learning. Omega. 2022;111:102643. doi: 10.1016/j.omega.2022.102643",
"flexflowsimcppo": "Alrashdan KR. FlexFlowSim-CPPO: simulator, protocol documents and archived results [Internet]. GitHub; 2026 [cited 2026 Sep 12]. Available from: https://github.com/alrashdank/flexflowsim-cppo",
"henderson2018matters": "Henderson P, Islam R, Bachman P, et al. Deep reinforcement learning that matters. In: Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence; 2018. p. 3207–3214. doi: 10.1609/aaai.v32i1.11694",
"huang2025evolving": "Huang Z, Mei Y, Zhang F, et al. Toward evolving dispatching rules with flow control operations by grammar-guided linear genetic programming. IEEE Trans Evol Comput. 2025;29(1):217–231. doi: 10.1109/TEVC.2024.3353207",
"li2025evolutionary": "Li C, Zhao X, Lin L, et al. An evolutionary knowledge training-based proximal policy optimization algorithm for job shop scheduling in flexible intelligent manufacturing. Comput Ind Eng. 2025;210:111533. doi: 10.1016/j.cie.2025.111533",
"liu2025gat": "Liu Y, Fan J, Shen W. A deep reinforcement learning approach with graph attention network and multi-signal differential reward for dynamic hybrid flow shop scheduling problem. J Manuf Syst. 2025;80:643–661. doi: 10.1016/j.jmsy.2025.03.028",
"marques2025dynamic": "Marques N, Figueira G, Guimarães L. Dynamic dispatching rule selection for the job shop scheduling problem. Comput Ind Eng. 2025;210:111471. doi: 10.1016/j.cie.2025.111471",
"mayerhoff2026slr": "Mayerhoff J, Schmidt M. Reinforcement learning for autonomous production planning and control: a systematic literature review. J Manuf Syst. 2026;86:546–568. doi: 10.1016/j.jmsy.2026.03.023",
"paternain2019duality": "Paternain S, Chamon LFO, Calvo-Fullana M, et al. Constrained reinforcement learning has zero duality gap. In: Advances in Neural Information Processing Systems 32; 2019. p. 7553–7563.",
"raffin2021sb3": "Raffin A, Hill A, Gleave A, et al. Stable-Baselines3: reliable reinforcement learning implementations. J Mach Learn Res. 2021;22(268):1–8.",
"rinciog2021fabricatio": "Rinciog A, Meyer A. Fabricatio-RL: a reinforcement learning simulation framework for production scheduling. In: Proceedings of the 2021 Winter Simulation Conference; 2021. p. 1–12. doi: 10.1109/WSC52266.2021.9715366",
"schneider2026role": "Schneider J, Pfannschmidt C, Nyhuis P, et al. The role of reinforcement learning in production control: a systematic literature review. IEEE Access. 2026;14:34375–34389. doi: 10.1109/ACCESS.2026.3668903",
"schulman2017ppo": "Schulman J, Wolski F, Dhariwal P, et al. Proximal policy optimization algorithms [preprint]. arXiv; 2017. arXiv:1707.06347. doi: 10.48550/arXiv.1707.06347",
"shen2026transformer": "Shen Y, Zhang X, Jin T. Transformer-based multi-agent reinforcement learning for flexible job shop scheduling with AGVs. Appl Soft Comput. 2026;193:114899. doi: 10.1016/j.asoc.2026.114899",
"stooke2020pid": "Stooke A, Achiam J, Abbeel P. Responsive safety in reinforcement learning by PID Lagrangian methods. In: Proceedings of the 37th International Conference on Machine Learning. PMLR 119; 2020. p. 9133–9143.",
"tang2020mask": "Tang CY, Liu CH, Chen WK, et al. Implementing action mask in proximal policy optimization (PPO) algorithm. ICT Express. 2020;6(3):200–203. doi: 10.1016/j.icte.2020.05.003",
"tessler2019rcpo": "Tessler C, Mankowitz DJ, Mannor S. Reward constrained policy optimization. In: 7th International Conference on Learning Representations; 2019.",
"wang2025end": "Wang R, Jing Y, Gu C, et al. End-to-end multitarget flexible job shop scheduling with deep reinforcement learning. IEEE Internet Things J. 2025;12(4):4420–4434. doi: 10.1109/JIOT.2024.3485748",
"zhang2025dual": "Zhang N, Liu B, Zhang J. Dual resource scheduling method of production equipment and rail-guided vehicles based on proximal policy optimization algorithm. Technologies. 2025;13(12):573. doi: 10.3390/technologies13120573",
}

src = pathlib.Path("IJMS_manuscript.md").read_text()
body = re.sub(r"^---\ntitle:.*?\n---\n", "", src, flags=re.S)
title = re.search(r'title: "(.*?)"', src).group(1)

# ---- anonymous copy for the submission portal ("Manuscript - anonymous") ----
# Removes the author block and everything that points at the author directly:
# the repository address (the GitHub account name identifies the author), the
# software reference, and the two sentences that say whose the initial study
# was.  The published self-citation stays in the third person, as the
# publisher's anonymous-review guidance allows.
ANON = "--anon" in sys.argv
SUFFIX = "_anon" if ANON else ""
if ANON:
    def sub1(old, new):
        global body
        assert body.count(old) == 1, old[:60]
        body = body.replace(old, new)
    body = re.sub(r"\A.*?(?=## Abstract)", "", body, flags=re.S)   # author block
    sub1("""the author's own earlier study, run under a protocol committed in advance, which""",
         """an earlier, unpublished study (details withheld for anonymous review), run
under a protocol committed in advance, which""")
    sub1("""The failure reported by the author's initial study of constrained reinforcement""",
         """The failure reported by the initial study of constrained reinforcement""")
    sub1("""here are openly available in the FlexFlowSim-CPPO repository at
https://github.com/alrashdank/flexflowsim-cppo, tag `ijms-submission` on branch
`ablation-slack-fix`.""",
         """here are openly available in a public repository at a tagged release; the
address is withheld for anonymous review and given in the version of this
manuscript that carries the author details.""")
    # Titles are withheld too: the simulator's name and the article title each
    # lead to the author in one web search.
    NLM["flexflowsimcppo"] = ("[Author, anonymised for review]. Simulator, protocol documents and "
                              "archived results of the study [Internet]; 2026. Name and repository "
                              "address withheld for anonymous review.")
    NLM["alrashdan2026breakdowns"] = ("[Author, anonymised for review]. Journal article introducing "
                                      "the two flow-shop instances and benchmarking dispatching rules, "
                                      "bandits and reinforcement learning under machine breakdowns; "
                                      "2026. Details withheld for anonymous review.")
    sub1("""**A.1 Software.** All experiments use the FlexFlowSim-CPPO simulator""",
         """**A.1 Software.** All experiments use the study's discrete-event simulator""")
    sub1("""On the two testbeds used here, Alrashdan
[@alrashdan2026breakdowns] benchmarked""",
         """On the two testbeds used here, an earlier study
[@alrashdan2026breakdowns] benchmarked""")
    body = re.sub(r"## Author contributions\n\n.*?\n\n(?=## )",
                  "## Author contributions\n\nWithheld for anonymous review.\n\n", body, flags=re.S)

order, seen = [], {}
def repl(m):
    keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
    nums = []
    for k in keys:
        if k not in NLM: raise SystemExit(f"unknown citation key: {k}")
        if k not in seen:
            order.append(k); seen[k] = len(order)
        nums.append(seen[k])
    return "[" + ",".join(str(x) for x in nums) + "]"
body = re.sub(r"\[((?:@[a-z0-9]+)(?:\s*;\s*@[a-z0-9]+)*)\]", repl, body)
assert "@" not in re.sub(r"\S+@\S+\.\S+", "", body), "unconverted citation remains"

reflist = "\n\n".join(f"{i}. {NLM[k]}" for i, k in enumerate(order, 1))
body = body.replace("## References\n", "## References\n\n" + reflist + "\n")

pathlib.Path(f"IJMS_manuscript_read{SUFFIX}.md").write_text(f'---\ntitle: "{title}"\n---\n' + body)

# ---- split into the running order the journal specifies -------------------
def cut(pat, text):
    m = re.search(pat, text, flags=re.S)
    return (m.group(0), text.replace(m.group(0), "")) if m else ("", text)

# pull tables (caption paragraph + pipe table) and figure blocks out of the body
tables = []
tpat = re.compile(r"\*\*Table ([A-B]?\d+)\.\*\*[\s\S]*?\n\n(?:\|[^\n]*\n)+")
for m in list(tpat.finditer(body)):
    tables.append((m.group(1), m.group(0).strip()))
    body = body.replace(m.group(0), f"[Table {m.group(1)} near here]\n\n")
figs = []
fpat = re.compile(r"!\[\]\((fig_\w+)\.png\)\{width=16cm\}\\\n\n\*\*Figure (\d)\.\*\*([\s\S]*?)(?=\n\n)")
for m in list(fpat.finditer(body)):
    figs.append((m.group(2), m.group(1), " ".join(m.group(3).split())))
    body = body.replace(m.group(0), f"[Figure {m.group(2)} near here]\n")

out = [f'---\ntitle: "{title}"\n---\n', body.rstrip(), "\n\\newpage\n\n## Tables\n"]
for i, (num, blk) in enumerate(tables):
    out.append(("\\newpage\n\n" if i else "") + blk + "\n")
# Journal running order: figures (one per page), then the captions as a list.
# The image description (alt text) goes into the docx for accessibility; the
# 600-dpi TIFF named in the caption list is the production file, supplied
# separately.
ALT = {"1": "Two-panel line chart of the throughput multiplier over training on "
            "a logarithmic axis, bakery left and electronics right; the original "
            "signal's five traces rise to the cap of 20,000, the corrected "
            "signal's five stay one to two orders of magnitude lower.",
       "2": "Two-panel chart of the symmetric cell over training: left, the "
            "throughput multiplier oscillating between 0 and about 1.5 around "
            "a dashed line at 0.1; right, the throughput slack per episode "
            "falling from about 4 towards 1 to 2 units above the floor."}
out.append("\\newpage\n\n## Figures\n")
for num, fname, cap in figs:
    out.append(("\\newpage\n\n" if num != figs[0][0] else "")
               + f"**Figure {num}**\n\n"
               f"![{ALT.get(num, '')}]({fname}.png){{width=16cm}}\\\n")
out.append("\\newpage\n\n## Figure captions\n")
for num, fname, cap in figs:
    out.append(f"**Figure {num}.** {cap} (file: Figure{num}.tif)\n")
# pandoc drops raw LaTeX when writing docx, so page breaks go in as raw OpenXML
PAGEBREAK = "```{=openxml}\n<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>\n```\n"
pathlib.Path(f"IJMS_manuscript_tf{SUFFIX}.md").write_text(
    "\n".join(out).replace("\\newpage\n\n", PAGEBREAK))
print(f"references: {len(order)} numbered; tables moved: {len(tables)}; figures moved: {len(figs)}")
print("citation order:", ", ".join(f"{i}={k}" for i, k in enumerate(order, 1)))
