"""Open the existing local demo or launch it with the selected Python runtime."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['auto','content','ufm'],default='auto')
    parser.add_argument('--data',type=Path,default=ROOT/'data/processed/toys_games_full_temporal')
    parser.add_argument('--run',type=Path,default=ROOT/'runs/ufm_full_v1')
    parser.add_argument('--features',type=Path)
    parser.add_argument('--training-source',type=Path)
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args(argv)
    if not 1<=args.port<=65535:
        parser.error('Port must be 1-65535.')
    ready=(args.run/'best.pt').is_file() and (args.run/'completed.json').is_file()
    backend=('ufm' if ready else 'content') if args.backend=='auto' else args.backend
    if backend=='ufm':
        required=[args.run/'best.pt',args.run/'config.json',args.run/'completed.json',
                  args.data/'content/items.csv.gz',args.data/'content/products_text.csv.gz']
        features=args.features or args.data/'foundation_clip_b32_v1'
        required += [features/name for name in ['complete.json','text.npy','image.npy','modalities.npy']]
        missing=[str(p) for p in required if not p.is_file()]
        if missing:
            print('UFM is not ready. Restore the verified backup first. Missing:')
            print('\n'.join(missing))
            return 1
        if args.training_source is None:
            archived=ROOT/'reports/fix_1_15_20261027/fitlab01_finish1/trained_snapshot'
            if (archived/'ufm_model.py').is_file():
                args.training_source=archived
    else:
        features=args.features
    url=f'http://127.0.0.1:{args.port}'
    try:
        with urllib.request.urlopen(url+'/api/status',timeout=2) as response:
            status = json.load(response)
        if status.get('app') != 'ufm-rec-demo':
            print(f'Port {args.port} belongs to another service. Choose --port with a free port.')
            return 1
        active_ufm=status.get('backend')=='UFM Rec'
        if active_ufm!=(backend=='ufm'):
            print(f'An existing demo uses {status.get("backend")}; requested {backend}. Choose another --port.')
            return 1
        webbrowser.open(url)
        return 0
    except urllib.error.URLError:
        pass
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        print('Missing dependencies. Run: py -m pip install -r requirements.txt')
        return 1
    command=[sys.executable,'-u',str(ROOT/'src/demo_recommender.py'),
             '--backend',backend,'--data',str(args.data),'--run',str(args.run),
             '--port',str(args.port),'--open-browser']
    if features is not None:
        command+=['--features',str(features)]
    if backend=='ufm' and args.training_source is not None:
        command+=['--training-source',str(args.training_source)]
    print(f'Selected backend: {backend}; data: {args.data}',flush=True)
    return subprocess.call(command,cwd=ROOT)


if __name__=='__main__':
    sys.exit(main())
