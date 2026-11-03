"""Refresh a Word document's indexes with LibreOffice and export it to PDF.

Run with the Python bundled in LibreOffice, which provides the ``uno`` module:

    <LibreOffice>/program/python.exe scripts/lo_update_export.py IN.docx OUT.pdf TOC.json PROFILE_DIR

TOC.json receives the refreshed text of each index (one "title<TAB>page" line
per entry), so the caller can cache the same page numbers inside the DOCX.
PROFILE_DIR holds a throwaway LibreOffice profile; the user's profile is untouched.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import uno
from com.sun.star.beans import PropertyValue


def prop(name, value):
    item = PropertyValue()
    item.Name = name
    item.Value = value
    return item


def connect(pipe):
    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
    for _ in range(240):
        try:
            return resolver.resolve(f"uno:pipe,name={pipe};urp;StarOffice.ComponentContext")
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("LibreOffice did not accept the UNO connection within 120 s")


def main():
    source, pdf, toc_json, profile = (Path(arg).resolve() for arg in sys.argv[1:5])
    office = Path(uno.__file__).resolve().parent / "soffice.exe"
    pipe = f"ufm_lo_{os.getpid()}"
    profile.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([str(office), "--headless", "--invisible", "--nologo", "--norestore",
                                "--nodefault", "--nolockcheck", f"-env:UserInstallation={profile.as_uri()}",
                                f"--accept=pipe,name={pipe};urp;"])
    desktop = None
    try:
        context = connect(pipe)
        desktop = context.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", context)
        document = desktop.loadComponentFromURL(source.as_uri(), "_blank", 0, (prop("Hidden", True),))
        indexes = document.getDocumentIndexes()
        # Update twice: a longer contents list can push later headings onto new pages.
        for _ in range(2):
            for i in range(indexes.getCount()):
                indexes.getByIndex(i).update()
        texts = [indexes.getByIndex(i).getAnchor().getString() for i in range(indexes.getCount())]
        document.storeToURL(pdf.as_uri(), (prop("FilterName", "writer_pdf_Export"),))
        document.close(True)
        toc_json.write_text(json.dumps(texts, ensure_ascii=False, indent=1), encoding="utf-8")
    finally:
        if desktop is not None:
            try:
                desktop.terminate()
            except Exception:
                pass
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    main()
