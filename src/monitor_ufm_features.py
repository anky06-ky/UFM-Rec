"""Read-only FITLAB progress; never starts/stops training or opens test samples."""
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/toys_games_full_temporal'


def main():
    status = ROOT / 'runs/foundation_queue_v1.json'
    print('QUEUE:', status.read_text(encoding='utf-8') if status.exists() else 'Not started')
    history = ROOT / 'runs/content_transformer_v1/history.jsonl'
    epochs = [json.loads(x) for x in history.read_text(encoding='utf-8').splitlines()] if history.exists() else []
    print('BASELINE COMPLETED EPOCHS:', len(epochs))
    if epochs:
        best = max(epochs, key=lambda x: x['validation']['known_user']['ndcg@10'])
        print('BEST BASELINE VALIDATION:', best['epoch'], best['validation']['known_user'])
    print('BASELINE COMPLETE:', (ROOT / 'runs/content_transformer_v1/completed.json').exists())
    catalog = DATA / 'foundation_catalog_v1/catalog_report.json'
    print('CATALOG:', json.loads(catalog.read_text(encoding='utf-8'))['counts'] if catalog.exists() else 'Missing')
    for name, folder in [('SMOKE', ROOT / 'runs/foundation_smoke_128_v1'), ('FULL', DATA / 'foundation_clip_b32_v1')]:
        complete, progress = folder / 'complete.json', folder / 'progress.json'
        if complete.exists():
            m = json.loads(complete.read_text(encoding='utf-8'))
            print(name, 'COMPLETE:', m['rows'], 'text:', m['has_text'], 'image:', m['has_image'])
        elif progress.exists():
            m = json.loads(progress.read_text(encoding='utf-8'))
            print(name, 'COMMITTED:', m['next_row'], '/', m['config']['rows'], m['image_status_counts'])
        else:
            print(name, 'NOT STARTED')
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv'], capture_output=True, text=True)
    print(result.stdout if result.returncode == 0 else 'nvidia-smi unavailable: ' + result.stderr)
    for name in ['content_transformer_v1.log', 'foundation_queue_v1.log']:
        path = ROOT / 'runs' / name
        if path.exists():
            print('LOG:', name)
            print(subprocess.run(['tail', '-n', '3', str(path)], capture_output=True, text=True).stdout)


if __name__ == '__main__':
    main()
