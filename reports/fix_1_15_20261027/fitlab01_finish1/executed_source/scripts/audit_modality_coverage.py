"""Report coverage of the verified CLIP cache, with item alignment and regime counts."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
from itertools import zip_longest
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import sha256, write_json
from build_recommender_samples import REGIMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new coverage report.')
    csv.field_size_limit(2**31 - 1)
    data = args.data
    features = data / 'foundation_clip_b32_v1'
    marker = json.loads((features / 'complete.json').read_text())
    flags = np.load(features / 'modalities.npy', mmap_mode='r')
    if sha256(features / 'modalities.npy') != marker['features_sha256']['modalities.npy']:
        raise ValueError('Modality mask checksum mismatch.')
    paths = [data / 'content/items.csv.gz', data / 'content/products_text.csv.gz',
             data / 'foundation_catalog_v1/catalog.csv.gz']
    totals = {name: dict(items=0, text=0, image=0, missing_text=0,
                        missing_image=0, missing_both=0, empty_image_url=0) for name in REGIMES}
    with gzip.open(paths[0], 'rt', encoding='utf-8') as a, \
         gzip.open(paths[1], 'rt', encoding='utf-8') as b, \
         gzip.open(paths[2], 'rt', encoding='utf-8') as c:
        count = 0
        for index, triple in enumerate(zip_longest(csv.DictReader(a), csv.DictReader(b), csv.DictReader(c))):
            item, text, image = triple
            if (any(row is None for row in triple) or int(item['row_index']) != index
                    or int(image['row_index']) != index
                    or len({row['parent_asin'] for row in triple}) != 1):
                raise ValueError(f'ASIN/row alignment failed at {index}.')
            n = int(item['train_count'])
            group = totals[REGIMES[0 if n == 0 else 1 if n <= 5 else 2 if n <= 20 else 3]]
            has_text, has_image = map(bool, flags[index + 1])
            group['items'] += 1
            group['text'] += int(has_text)
            group['image'] += int(has_image)
            group['missing_text'] += int(not has_text)
            group['missing_image'] += int(not has_image)
            group['missing_both'] += int(not (has_text or has_image))
            group['empty_image_url'] += int(not image.get('image_url', '').strip())
            count += 1
    if flags.shape != (count + 1, 2) or count != marker['rows']:
        raise ValueError('Mask/marker/catalog row count mismatch.')
    report = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        items=count, by_regime=totals, image_status_counts=marker.get('image_status_counts'),
        sources={str(p.relative_to(data)): sha256(p) for p in paths + [features / 'complete.json', features / 'modalities.npy']},
        feature_hashes=marker['features_sha256'], text_max_tokens=marker.get('text_max_tokens', 77),
        limitations=['Metadata is a snapshot and can include edits after interaction time.',
                     'CLIP text is truncated to 77 tokens.',
                     'Failure reasons are retained globally by the extractor; no per-item failure log is available here to assign each reason by regime.',
                     'URL availability is not re-downloaded; image coverage describes the existing trained cache.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(dict(passed=True, items=count, by_regime=totals)), flush=True)


if __name__ == '__main__':
    main()
