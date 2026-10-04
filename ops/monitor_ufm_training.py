"""Read-only UFM pipeline status; standard library, no kernel/GPU allocations."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    for relative in ['runs/ufm_training_queue_v1.json','runs/foundation_queue_v1.json',
                     'data/processed/ufm_positive_graph_h20_v1/complete.json',
                     'data/processed/toys_games_full_temporal/foundation_clip_b32_v1/progress.json',
                     'runs/ufm_gpu_smoke_v1/completed.json','runs/ufm_full_v1/completed.json']:
        p = ROOT/relative
        print('\n'+relative)
        if not p.exists():
            print('NOT CREATED'); continue
        state = json.loads(p.read_text())
        if relative.endswith('progress.json'):
            state = {k:state[k] for k in ['next_row','image_status_counts']}
        elif 'positive_graph' in relative:
            state = {k:state[k] for k in ['rows','items','users','edges','policy']}
        print(json.dumps(state,ensure_ascii=False,indent=2))
    epochs = sorted((ROOT/'runs/ufm_full_v1').glob('epoch_*.json'))
    print('\nUFM COMPLETED EPOCHS:', len(epochs))
    if epochs:
        print(json.dumps(json.loads(epochs[-1].read_text()),indent=2))
    for relative in ['runs/ufm_training_queue_v1.log','runs/foundation_queue_v1.log']:
        p = ROOT/relative
        if p.exists():
            with p.open('rb') as source:
                source.seek(max(0,p.stat().st_size-4096))
                lines = source.read().decode('utf-8',errors='replace').splitlines()
            print('\nLOG:', relative)
            print('\n'.join(lines[-4:]))
    subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv'],check=True)


if __name__ == '__main__':
    main()
