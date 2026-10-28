"""Convert a LaTeX display formula into native Word OMML.

Pandoc ships a LaTeX math reader that emits OMML directly for docx output,
so the pipeline is: LaTeX -> pandoc -> docx -> extract <m:oMathPara>.

Usage:
    python scripts/latex_to_omml.py "a^2 + b^2 = c^2"
"""

import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PANDOC = Path(os.environ.get("PANDOC_BIN", "")) if os.environ.get("PANDOC_BIN") else None


def _find_pandoc():
    global PANDOC
    if PANDOC and PANDOC.exists():
        return PANDOC
    files = BASE_DIR / "tmp" / "omml_tools" / "pypandoc" / "files"
    for candidate in (files / "pandoc.exe", files / "pandoc"):
        if candidate.exists():
            PANDOC = candidate
            return PANDOC
    try:
        sys.path.insert(0, str(BASE_DIR / "tmp" / "omml_tools"))
        import pypandoc
        PANDOC = Path(pypandoc.get_pandoc_path())
        if PANDOC.exists():
            return PANDOC
    except Exception:
        pass
    from shutil import which
    found = which("pandoc")
    if not found:
        raise RuntimeError("pandoc not found; install pypandoc_binary or set PANDOC_BIN.")
    PANDOC = Path(found)
    return PANDOC


def latex_to_omml(latex: str) -> str:
    """Return the <m:oMathPara> XML for a LaTeX display formula."""
    pandoc = _find_pandoc()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = tmp / "formula.md"
        source.write_text(f"$$\n{latex}\n$$\n", encoding="utf-8")
        output = tmp / "formula.docx"
        result = subprocess.run(
            [str(pandoc), "-f", "markdown+tex_math_dollars", "-t", "docx", "-o", str(output), str(source)],
            capture_output=True, text=True,
        )
        if result.returncode or not output.exists():
            raise RuntimeError(f"pandoc failed: {result.stderr.strip()}")
        with zipfile.ZipFile(output) as archive:
            xml = archive.read("word/document.xml").decode("utf-8")
    match = re.search(r"<m:oMathPara\b.*?</m:oMathPara>", xml, re.S)
    if not match:
        raise RuntimeError(f"No oMathPara produced for: {latex[:80]}")
    return match.group(0)


if __name__ == "__main__":
    print(latex_to_omml(sys.argv[1]))
