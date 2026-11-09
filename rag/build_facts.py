"""Product facts for the RAG layer (price, rating, category, age, short bullets).

Source: raw Amazon Reviews 2023 metadata (Toys_and_Games). Output is aligned with
content/items.csv.gz row_index. These facts are a 2023 metadata snapshot: they are shown to
the user and used as hard filters in the demo, never as features in the offline benchmark.
"""
import csv, gzip, json, re, sys
from pathlib import Path
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/toys_games_full_temporal'
RAW = ROOT / 'data/raw/amazon_reviews_2023/raw_meta_Toys_and_Games'
OUT = DATA / 'rag_facts_v1'


def age_years(details):
    """Minimum manufacturer age in years, or nan."""
    try:
        d = json.loads(details) if details else {}
    except ValueError:
        return np.nan
    s = str(d.get('Manufacturer recommended age') or '')
    m = re.search(r'(\d+(?:\.\d+)?)\s*(month|year)', s, re.I)
    if not m:
        return np.nan
    v = float(m.group(1))
    return v / 12 if m.group(2).lower().startswith('month') else v


def price(s):
    try:
        v = float(str(s).replace('$', '').replace(',', ''))
        return v if 0 < v < 100000 else np.nan
    except ValueError:
        return np.nan


def clip(s, n):
    s = re.sub(r'\s+', ' ', str(s or '')).strip()
    return s if len(s) <= n else s[:n - 1].rsplit(' ', 1)[0] + '…'


def main():
    lookup = {}
    with gzip.open(DATA / 'content/items.csv.gz', 'rt', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            lookup[r['parent_asin']] = int(r['row_index'])
    n = len(lookup)
    cols = dict(price=np.full(n, np.nan, np.float32), rating=np.full(n, np.nan, np.float32),
                rating_number=np.zeros(n, np.int32), age_min=np.full(n, np.nan, np.float32))
    text = {k: [''] * n for k in ['category', 'store', 'bullets', 'description']}
    seen = 0
    for path in sorted(RAW.glob('*.parquet')):
        f = pq.ParquetFile(path)
        for b in f.iter_batches(batch_size=50000, columns=['parent_asin', 'price', 'average_rating', 'rating_number',
                                                          'categories', 'store', 'features', 'description', 'details']):
            for r in b.to_pylist():
                i = lookup.get(r['parent_asin'])
                if i is None:
                    continue
                seen += 1
                cols['price'][i] = price(r['price'])
                if r['average_rating'] is not None:
                    cols['rating'][i] = r['average_rating']
                cols['rating_number'][i] = r['rating_number'] or 0
                cols['age_min'][i] = age_years(r['details'])
                cats = [c for c in (r['categories'] or []) if c != 'Toys & Games']
                text['category'][i] = clip(' > '.join(cats[-2:]), 80)
                text['store'][i] = clip(r['store'], 60)
                text['bullets'][i] = ' | '.join(clip(x, 160) for x in (r['features'] or [])[:3])
                text['description'][i] = clip(' '.join(r['description'] or []), 400)
        print(path.name, seen, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    table = pa.table({'row_index': np.arange(n, dtype=np.int32), **cols, **text})
    pq.write_table(table, OUT / 'facts.parquet', compression='zstd')
    report = dict(items=n, matched=seen, price_known=int(np.isfinite(cols['price']).sum()),
                  age_known=int(np.isfinite(cols['age_min']).sum()),
                  note='2023 metadata snapshot; display and demo filters only, not benchmark features.')
    (OUT / 'complete.json').write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == '__main__':
    main()
