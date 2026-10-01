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
    progress_path=features/'progress.json'
    progress_age=(datetime.now(timezone.utc).timestamp()-progress_path.stat().st_mtime
                  if progress_path.is_file() else None)
    total=complete.get('rows') or progress.get('config',{}).get('rows',767045)
    rows=complete.get('rows') or progress.get('next_row',0)
    recovery=read(root/'runs/campaign_recovery_20261001.json')
    if recovery and isinstance(recovery.get('pid'),int):
        recovery['supervisor_alive']=(Path('/proc')/str(recovery['pid'])).exists()
    watchdog=read(root/'runs/campaign_watchdog_v1.json')
    if watchdog and isinstance(watchdog.get('pid'),int):
        watchdog['watchdog_alive']=(Path('/proc')/str(watchdog['pid'])).exists()
    state=dict(time_utc=datetime.now(timezone.utc).isoformat(),
        clip=dict(rows=rows,total=total,percent=round(100*rows/total,2) if total else 0,
                  complete=bool(complete and 'read_error' not in complete),
                  image_ok=complete.get('has_image',progress.get('image_status_counts',{}).get('ok',0)),
                  progress_age_seconds=round(progress_age) if progress_age is not None else None),
        foundation=read(root/'runs/foundation_queue_v1.json'),
        ufm=read(root/'runs/ufm_training_queue_v1.json'),
        ablations=read(root/'runs/ufm_ablation_suite_v1.json'),
        recovery=recovery,
        watchdog=watchdog,
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
    for key,label in [('watchdog','WATCHDOG'),('recovery','RECOVERY'),('foundation','FOUNDATION'),
                      ('ufm','UFM'),('ablations','ABLATION')]:
        value=state[key] or {}
        if (key in ('ufm','ablations') and value.get('stage')=='stopped_with_error'
                and (state['recovery'] or {}).get('supervisor_alive')
                and (state['recovery'] or {}).get('stage')=='foundation_running'):
            print(label+': queued_after_clip | previous attempt stopped')
            continue
        heartbeat=value.get('time_utc')
        active=(key=='foundation' and value.get('stage')=='full_extraction_running'
                and clip['progress_age_seconds'] is not None
                and clip['progress_age_seconds']<=300)
        age=''
        if heartbeat:
            try:
                seconds=(datetime.now(timezone.utc)-datetime.fromisoformat(heartbeat)).total_seconds()
                if seconds>300 and not value.get('supervisor_alive') and not active:
                    age=f' | stale {seconds/3600:.1f}h'
            except ValueError:
                age=' | invalid timestamp'
        alive=' | supervisor alive' if value.get('supervisor_alive') else ''
        if value.get('watchdog_alive'):alive+=' | watchdog alive'
        if active:alive+=' | progress active'
        print(label+': '+value.get('stage','Chua co trang thai')+alive+age)
        if heartbeat:print('  Heartbeat: '+heartbeat)
        if value.get('gpu'):print('  GPU: '+json.dumps(value['gpu']))
    print('FULL CHECKPOINT: '+('complete' if state['full_complete'] else 'not complete'))
    print('NEXT: docs/next.md | LOGS: runs/ | GUIDE: START.md')


if __name__=='__main__':main()
