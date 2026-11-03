"""Refresh a Word document's indexes with LibreOffice and export it to PDF.

Run with the Python bundled in LibreOffice, which provides the ``uno`` module:

    <LibreOffice>/program/python.exe scripts/lo_update_export.py IN.docx OUT.pdf LAYOUT.json PROFILE_DIR [CAPTION_STYLE]

LAYOUT.json receives the refreshed text of each index ("title<TAB>page" lines),
the physical page of every paragraph in CAPTION_STYLE, and the physical pages of
the first front-matter and first chapter headings, so the caller can cache the
same page numbers inside the DOCX. PROFILE_DIR holds a throwaway profile.
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


def layout(document, caption_style):
    """Physical page of caption paragraphs and of the two anchor headings."""
    cursor = document.getCurrentController().getViewCursor()
    captions, anchors = [], {}
    paragraphs = document.getText().createEnumeration()
    while paragraphs.hasMoreElements():
        paragraph = paragraphs.nextElement()
        if not paragraph.supportsService("com.sun.star.text.Paragraph"):
            continue  # tables sit between paragraphs at this level
        style, text = paragraph.ParaStyleName, paragraph.getString().strip()
        if style != caption_style and not style.startswith("Heading"):
            continue
        cursor.gotoRange(paragraph.getStart(), False)
        if style == caption_style:
            captions.append({"text": text, "page": cursor.getPage()})
        elif text.startswith("LỜI CẢM ƠN"):
            anchors.setdefault("front", cursor.getPage())
        elif text.startswith("CHƯƠNG 1."):
            anchors.setdefault("body", cursor.getPage())
    return captions, anchors


def main():
    source, pdf, layout_json, profile = (Path(arg).resolve() for arg in sys.argv[1:5])
    caption_style = sys.argv[5] if len(sys.argv) > 5 else ""
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
        captions, anchors = layout(document, caption_style) if caption_style else ([], {})
        document.storeToURL(pdf.as_uri(), (prop("FilterName", "writer_pdf_Export"),))
        document.close(True)
        layout_json.write_text(json.dumps({"indexes": texts, "captions": captions, "anchors": anchors},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
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
