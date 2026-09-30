"""Read-only campaign overview; standard library only, no GPU allocation."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def read(path):
    if not path.is_file():return None
    try:return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):return {'read_error':str(path)}


def snapshot(root):
    features=root/'data/processed/toys_games_full_temporal/foundation_clip_b32_v1'
    progress,complete=read(features/'progress.json'),read(features/'complete.json')
    progress=progress or {};complete=complete or {}
    total=complete.get('rows') or progress.get('config',{}).get('rows',767045)
    rows=complete.get('rows') or progress.get('next_row',0)
    state=dict(time_utc=datetime.now(timezone.utc).isoformat(),
        clip=dict(rows=rows,total=total,percent=round(100*rows/total,2) if total else 0,
                  complete=bool(complete and 'read_error' not in complete),
                  image_ok=complete.get('has_image',progress.get('image_status_counts',{}).get('ok',0))),
        ufm=read(root/'runs/ufm_training_queue_v1.json'),
        ablations=read(root/'runs/ufm_ablation_suite_v1.json'),
        full_complete=read(root/'runs/ufm_full_v1/completed.json'))
    return state


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--json',action='store_true')
    args=p.parse_args(argv);state=snapshot(args.root)
    if args.json:print(json.dumps(state,ensure_ascii=False,indent=2));return
    clip=state['clip']
    print('UFM REC | '+state['time_utc'])
    print(f"CLIP: {clip['rows']:,}/{clip['total']:,} ({clip['percent']:.2f}%) | images OK: {clip['image_ok']:,}")
    for key,label in [('ufm','UFM'),('ablations','ABLATION')]:
        value=state[key] or {}
        print(label+': '+value.get('stage','Chua co trang thai'))
        if value.get('time_utc'):print('  Heartbeat: '+value['time_utc'])
        if value.get('gpu'):print('  GPU: '+json.dumps(value['gpu']))
    print('FULL CHECKPOINT: '+('complete' if state['full_complete'] else 'not complete'))
    print('NEXT: docs/next.md | LOGS: runs/ | GUIDE: START.md')


if __name__=='__main__':main()
