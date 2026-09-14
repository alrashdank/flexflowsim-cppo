#!/usr/bin/env python3
"""Render the two figure PDFs to 600-dpi LZW TIFFs for submission.

    python3 make_tiffs.py            # writes /home/claude/draft/Figure1.tif, Figure2.tif

Requires pdftoppm (poppler) and Pillow.  The PDFs are drawn by make_fig_lambda.py
(Figure 1) and make_fig_r2.py (Figure 2)."""
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

DRAFT = Path("/home/claude/draft")
FIGS = {"Figure1": "fig_lambda_T.pdf", "Figure2": "fig_symmetric.pdf"}

for name, pdf in FIGS.items():
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdftoppm", "-r", "600", "-png", "-singlefile",
                        str(DRAFT / pdf), f"{td}/{name}"], check=True)
        im = Image.open(f"{td}/{name}.png").convert("RGB")
        im.save(DRAFT / f"{name}.tif", compression="tiff_lzw", dpi=(600, 600))
        print(name, im.size, "600 dpi LZW")
