"""Checks for the school-template report builder that need neither Pandoc nor LibreOffice."""
import re
import sys
from pathlib import Path

import pytest

docx = pytest.importorskip("docx")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_school_report_docx as report  # noqa: E402


def test_every_table_and_figure_gets_a_numbered_caption():
    source = report.SOURCE.read_text(encoding="utf-8")
    markdown = report.report_markdown()
    tables = re.findall(r"^@@TABLE@@Bảng (\d+)\.(\d+)\. \S", markdown, re.M)
    figures = re.findall(r"^@@FIGURE@@Hình (\d+)\.(\d+)\. \S", markdown, re.M)
    assert len(tables) == len(re.findall(r"^\|\s*:?-", source, re.M))
    assert len(figures) == len(re.findall(r"^!\[", source, re.M))
    for numbers in (tables, figures):
        for chapter in {c for c, _ in numbers}:
            assert [int(n) for c, n in numbers if c == chapter] == list(
                range(1, sum(c == chapter for c, _ in numbers) + 1))


def test_repository_links_become_printable_paths():
    markdown = report.report_markdown()
    assert "](../" not in markdown
    assert "(`reports/continuation_20261028/validation_v3_results.json`)" in markdown
    assert "](https://arxiv.org/abs/2103.00020)" in markdown


def test_cover_line_rewrite_keeps_label_and_value_runs():
    paragraph = docx.Document().add_paragraph()
    paragraph.add_run("")  # Word-saved covers start some lines with an empty run
    paragraph.add_run("SVTH 1:  ").bold = True
    paragraph.add_run("Tên cũ – MSSV: 1")
    report.set_texts(paragraph, "SVTH 1:  ", "Nguyễn Văn A – MSSV: 2")
    assert paragraph.text == "SVTH 1:  Nguyễn Văn A – MSSV: 2"
    assert [run.bold for run in paragraph.runs] == [True, None]
