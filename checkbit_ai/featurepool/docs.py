"""Regenerate docs/FEATURES.md from the feature definitions:  python -m checkbit_ai.featurepool.docs"""
from pathlib import Path

from .definitions import render_markdown

DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "FEATURES.md"

if __name__ == "__main__":
    DOC_PATH.write_text(render_markdown(), encoding="utf-8")
    print(f"wrote {DOC_PATH}")
