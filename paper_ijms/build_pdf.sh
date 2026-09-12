#!/usr/bin/env bash
# Reading copy: numbered references, tables and figures in place, equations rendered.
# The .docx built by build_ijms.sh is the submission file; this PDF is for reading only.
set -e; cd "$(dirname "$0")"
python3 refs_nlm.py > /dev/null
sed 's/{width=16cm}\\$/{width=100%}/' IJMS_manuscript_read.md > /tmp/_read.md
pandoc -f markdown+tex_math_single_backslash /tmp/_read.md -o IJMS_manuscript.pdf \
  --pdf-engine=xelatex --resource-path=. \
  -V mainfont="TeX Gyre Termes" -V mathfont="Latin Modern Math" \
  -V geometry:margin=2.5cm -V fontsize=11pt -V linestretch=1.15 \
  -V colorlinks=false
