#!/usr/bin/env python3
"""Build the TMLR supplementary zip (S1 + anonymised S2 code/results snapshot) from a clone of this
repository. Run from the repository root: python3 paper/make_supplementary.py path/to/S1_original_manuscript.pdf
Output: TMLR_supplementary.zip (about 60 MB; TMLR limit 100 MB)."""
import re, sys, shutil, subprocess, pathlib, json, glob, zipfile, tempfile
S1 = pathlib.Path(sys.argv[1]).resolve()
root = pathlib.Path(".").resolve(); tmp = pathlib.Path(tempfile.mkdtemp()); snap = tmp / "flexflowsim-cppo-anon"; snap.mkdir()
subprocess.run(f"git archive HEAD | tar -x -C {snap}", shell=True, check=True)
# scrub identifiers
(snap/"LICENSE").write_text(re.sub(r"Copyright \(c\) 2026 .*", "Copyright (c) 2026 The Authors (anonymised for review)", (snap/"LICENSE").read_text()))
for f in ["README.md", "paper6_v6_pilots_README.md", "protocol.md"]:
    p = snap / f
    if not p.exists(): continue
    s = p.read_text()
    s = re.sub(r"\(Alrashdan[^)]*\)", "(Author, 2026)", s).replace("Alrashdan, K. R. (2026)", "Author (2026)")
    s = s.replace("**Author:** Khaled R. Alrashdan", "**Author:** [anonymised for review]").replace("(Khaled, 2026-05-11)", "(author, 2026-05-11)")
    s = re.sub(r"https://github\.com/alrashdank/[\w\-]+", "[repository URL withheld for review]", s)
    p.write_text(s)
# untracked result files: everything except checkpoints, plus the selected checkpoint of every run
for r in ["results_r1", "results_r2", "results_ablation"]:
    for f in pathlib.Path(r).rglob("*"):
        if f.is_file() and "checkpoints" not in f.parts and f.suffix != ".zip":
            (snap / f.parent).mkdir(parents=True, exist_ok=True); shutil.copy(f, snap / f)
for p in glob.glob("results_r1/*/*/seed_*/summary.json") + glob.glob("results_r2/*/*/seed_*/summary.json") + glob.glob("results_ablation/*/*/seed_*/summary.json"):
    s = json.load(open(p))
    if "selected_steps" in s:
        ck = pathlib.Path(p).parent / "checkpoints" / f"ckpt_{s['selected_steps']}_steps.zip"
        if ck.exists(): (snap / ck.parent).mkdir(parents=True, exist_ok=True); shutil.copy(ck, snap / ck)
shutil.copy(root / "paper" / "README_SUPPLEMENT.md", snap / "README_SUPPLEMENT.md")
bad = subprocess.run(["grep", "-rli", r"alrashdan\|khaled\|paaet", str(snap)], capture_output=True, text=True).stdout.strip()
assert not bad, f"identifiers remain in: {bad}"
s2 = tmp / "S2_code_and_results.zip"
with zipfile.ZipFile(s2, "w", zipfile.ZIP_DEFLATED) as z:
    for f in snap.rglob("*"):
        if f.is_file() and "__pycache__" not in f.parts: z.write(f, f.relative_to(tmp))
with zipfile.ZipFile("TMLR_supplementary.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(root / "paper" / "README_supplementary.txt", "README_supplementary.txt"); z.write(S1, "S1_original_manuscript.pdf"); z.write(s2, "S2_code_and_results.zip")
print("wrote TMLR_supplementary.zip", round(pathlib.Path("TMLR_supplementary.zip").stat().st_size / 1e6, 1), "MB")
