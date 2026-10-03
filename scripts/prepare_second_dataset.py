"""Prepare Amazon 2023 All_Beauty with the existing temporal protocol, CPU only.

Official source: https://amazon-reviews-2023.github.io/
Creates independent paths; never writes Toys & Games files or evaluates test.
Completed stages are reused. Partial preprocessing stages require inspection.
"""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
RAW = ROOT/'data/raw/all_beauty_2023'
DATA = ROOT/'data/processed/all_beauty_full_temporal'
URL = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = RAW/'sources.json'
    sources = json.loads(manifest.read_text()) if manifest.exists() else {}
    for kind, name in [('review_categories','All_Beauty.jsonl.gz'),
                       ('meta_categories','meta_All_Beauty.jsonl.gz')]:
        path = RAW/name
        url = URL+kind+'/'+name
        if path.exists():
            if name not in sources or sources[name]['sha256'] != digest(path):
                raise ValueError('Existing download has no matching source hash: '+name)
        else:
            temporary = path.with_suffix(path.suffix+'.part')
            print('Downloading '+url, flush=True)
            with urllib.request.urlopen(url, timeout=60) as source, temporary.open('wb') as target:
                size = 0
                while block := source.read(1024*1024):
                    target.write(block); size += len(block)
                expected = source.headers.get('Content-Length')
                if expected is not None and size != int(expected):
                    raise IOError('Incomplete download: '+name)
            with temporary.open('rb') as f:
                if f.read(2) != b'\x1f\x8b':
                    raise ValueError('Expected gzip dataset, not an HTML error page.')
            temporary.replace(path)
            sources[name] = dict(url=url, bytes=size, sha256=digest(path))
            manifest.write_text(json.dumps(sources, indent=2), encoding='utf-8')
    import prepare_full_temporal as temporal
    temporal.RAW, temporal.METADATA = RAW/'All_Beauty.jsonl.gz', RAW/'meta_All_Beauty.jsonl.gz'
    temporal.OUTPUT, temporal.WORK = DATA, ROOT/'tmp/prepare_all_beauty_full_temporal'
    if not (DATA/'split_manifest.json').exists():
        temporal.main()
    import build_full_content_features as content
    content.DATA, content.METADATA = DATA, temporal.METADATA
    content.OUTPUT, content.PRODUCTS = DATA/'content', DATA/'content/products_text.csv.gz'
    if not (content.OUTPUT/'content_report.json').exists():
        content.main()
    import build_recommender_samples as samples
    samples.DATA, samples.CONTENT, samples.OUTPUT = DATA, DATA/'content', DATA/'evaluation'
    if not (samples.OUTPUT/'sampling_report.json').exists():
        samples.main()
    from prepare_ufm_catalog import build_catalog
    if not (DATA/'foundation_catalog_v1/catalog_report.json').exists():
        build_catalog(DATA, temporal.METADATA, DATA/'foundation_catalog_v1')
    import evaluate_content_baseline as baseline
    baseline.DATA, baseline.CONTENT = DATA, DATA/'content'
    baseline.EVALUATION, baseline.OUTPUT = DATA/'evaluation', DATA/'models/content_baseline'
    baseline.SPLITS = ('validation',)
    if not (baseline.OUTPUT/'metrics.json').exists():
        baseline.main()
    from train_gpu_recommender import prepare
    prepare(DATA, ROOT/'data/processed/all_beauty_gpu_cache_h20_t64_v1', 20, 64)
    report = dict(dataset='Amazon Reviews 2023 All_Beauty', sources=sources,
                  prepared_data=str(DATA), test_evaluated=False,
                  remaining=['CLIP extraction and feature coverage audit', 'UFM/baseline training and comparison'])
    (ROOT/'reports/second_dataset_preparation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SECOND_DATASET_CPU_PREPARATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
