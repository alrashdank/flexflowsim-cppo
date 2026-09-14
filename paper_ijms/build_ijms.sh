#!/usr/bin/env bash
# Build the IJMS submission Word file from IJMS_manuscript_tf.md
set -e; cd "$(dirname "$0")"
python3 refs_nlm.py > /dev/null
pandoc -f markdown+tex_math_single_backslash IJMS_manuscript_tf.md -o IJMS_manuscript.docx --reference-doc=reference.docx --resource-path=.
python3 postprocess_docx.py IJMS_manuscript.docx
# Second Word file for readers: tables and figures in place, as in the PDF.
pandoc -f markdown+tex_math_single_backslash IJMS_manuscript_read.md -o IJMS_manuscript_inline.docx --reference-doc=reference.docx --resource-path=.
python3 postprocess_docx.py IJMS_manuscript_inline.docx
# Anonymous copy required by the submission portal ("Manuscript - anonymous").
python3 refs_nlm.py --anon > /dev/null
pandoc -f markdown+tex_math_single_backslash IJMS_manuscript_tf_anon.md -o IJMS_manuscript_anonymous.docx --reference-doc=reference.docx --resource-path=.
python3 postprocess_docx.py IJMS_manuscript_anonymous.docx
