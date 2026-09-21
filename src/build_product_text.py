"""Build product text from split metadata without fitting a model."""

import csv
import gzip
import html
import json
import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path


DATA = Path(__file__).resolve().parents[1] / "data/processed/toys_games_70_15_15"
SPLITS = ("train", "validation", "test")
FIELDS = ("title", "features", "description")
INVISIBLE = str.maketrans("", "", "\u200b\u200c\u200d\u2060\ufeff")


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1
        if tag in ("br", "p", "div", "li", "tr", "td", "h1", "h2"):
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden = max(0, self.hidden - 1)
        if tag in ("p", "div", "li", "tr", "td", "h1", "h2"):
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(part for item in value if (part := clean_text(item)))
    if not isinstance(value, str):
        raise TypeError(f"Expected string/list/null; found {type(value).__name__}")
    if re.search(r"</?[a-zA-Z][^>]*>", value):
        parser = VisibleText()
        parser.feed(value)
        parser.close()
        value = "".join(parser.parts)
    value = html.unescape(value)
    value = unicodedata.normalize("NFC", value).translate(INVISIBLE)
    return " ".join(value.split())


def self_check():
    assert clean_text(None) == clean_text([]) == ""
    assert clean_text(["  Car\u200b", None, "\n  Ages 3–8\xa0"]) == "Car Ages 3–8"
    assert clean_text("<p>Red <b>car</b></p><p>A &amp; B</p>") == "Red car A & B"
    assert clean_text("Car<script>alert(1)</script><style>x</style> toy") == "Car toy"
    assert clean_text("Price < $20; 1:24; 4WD") == "Price < $20; 1:24; 4WD"


def main():
    self_check()
    with gzip.open(DATA / "item_split.csv.gz", "rt", encoding="utf-8", newline="") as f:
        expected = {split: set() for split in SPLITS}
        for row in csv.DictReader(f):
            expected[row["split"]].add(row["parent_asin"])

    report = {
        "source_fields": list(FIELDS),
        "cleaning": "Remove HTML markup/script/style, decode entities, normalize NFC, remove selected invisible characters, collapse whitespace.",
        "text_order": "title, features, description",
        "policy": "Preserve language, case, numbers and punctuation; no truncation or model fitting; retain and report empty text.",
        "splits": {},
    }
    for split in SPLITS:
        print(f"Processing {split}...", flush=True)
        target = DATA / f"products_text_{split}.csv.gz"
        temporary = target.with_suffix(target.suffix + ".tmp")
        seen = set()
        stats = {"products": 0, "empty_text": 0, "empty_title": 0,
                 "empty_features": 0, "empty_description": 0,
                 "total_characters": 0, "max_characters": 0}
        with gzip.open(DATA / f"metadata_{split}.jsonl.gz", "rt", encoding="utf-8") as source, \
                gzip.open(temporary, "wt", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=["parent_asin", "title", "text"])
            writer.writeheader()
            for line in source:
                product = json.loads(line)
                item = product["parent_asin"]
                assert item in expected[split] and item not in seen, item
                seen.add(item)
                cleaned = {field: clean_text(product.get(field)) for field in FIELDS}
                text = " ".join(part for part in cleaned.values() if part)
                writer.writerow({"parent_asin": item, "title": cleaned["title"], "text": text})
                stats["products"] += 1
                stats["empty_text"] += int(not text)
                for field in FIELDS:
                    stats[f"empty_{field}"] += int(not cleaned[field])
                stats["total_characters"] += len(text)
                stats["max_characters"] = max(stats["max_characters"], len(text))
        assert seen == expected[split], f"Missing products in {split}"

        # Reopen the serialized output and check IDs, row counts and whitespace.
        reopened = set()
        with gzip.open(temporary, "rt", encoding="utf-8", newline="") as output:
            reader = csv.DictReader(output)
            assert reader.fieldnames == ["parent_asin", "title", "text"]
            for row in reader:
                assert row["parent_asin"] not in reopened
                reopened.add(row["parent_asin"])
                assert row["text"] == " ".join(row["text"].split())
                assert row["text"] == row["text"].translate(INVISIBLE)
        assert reopened == seen
        temporary.replace(target)
        stats["mean_characters"] = round(stats["total_characters"] / stats["products"], 2)
        report["splits"][split] = stats
        print(f"  {stats['products']:,} products; {stats['empty_text']:,} empty texts", flush=True)

    temporary_report = DATA / "product_text_report.json.tmp"
    temporary_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_report.replace(DATA / "product_text_report.json")
    print("PASS: all product text files created and verified.")


if __name__ == "__main__":
    main()
