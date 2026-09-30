"""Organize documentation/notebooks only; dry run unless --apply is supplied."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile

RENAMES={
 'docs/UFM_FULL_SCOPE.md':'docs/scope.md',
 'docs/GPU_TRAINING.md':'docs/gpu.md',
 'docs/FITLAB.md':'docs/fitlab.md',
 'docs/FITLAB_FOUNDATION_20260928.md':'docs/foundation.md',
 'docs/FITLAB_UFM_TRAINING_20260928.md':'docs/train.md',
 'docs/LONG_RUNNING_EXPERIMENTS.md':'docs/ops.md',
 'docs/FITLAB_RECOVERY_20260928.md':'docs/recover.md',
 'docs/DEMO_GUIDE.md':'docs/demo.md',
 'docs/de_xuat_do_an.html':'docs/proposal.html',
 'docs/tom_tat_du_an.html':'docs/overview.html',
 'notebooks/FITLAB_Start.ipynb':'notebooks/01_setup.ipynb',
 'notebooks/FITLAB_Training_Monitor.ipynb':'notebooks/02_baseline.ipynb',
 'notebooks/FITLAB_UFM_Features_Monitor.ipynb':'notebooks/03_clip.ipynb',
}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def within(root,relative):
    path=root/relative
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('Unsafe target: '+relative)
    return path


def plan(root):
    root=root.resolve();moves=[]
    existing={p.relative_to(root).as_posix() for folder in ['docs','notebooks'] for p in (root/folder).iterdir()}
    for old,new in RENAMES.items():
        source,target=within(root,old),within(root,new)
        if old in existing:
            same_case_path=old.lower()==new.lower() and target.exists() and source.samefile(target)
            if target.exists() and not same_case_path:raise FileExistsError('Rename collision: '+new)
            moves.append((old,new))
    for folder in [root,root/'docs']:
        for source in sorted(folder.iterdir()):
            if not source.is_file():continue
            if source.suffix=='.tar':new='archive/transfers/'+source.name
            elif folder==root/'docs' and 'before_' in source.name:new='archive/notes/'+source.name
            else:continue
            old=source.relative_to(root).as_posix()
            within(root,old)
            if within(root,new).exists():raise FileExistsError('Archive collision: '+new)
            moves.append((old,new))
    changes={}
    documents=[root/'README.md',*(root/'docs').glob('*.md'),*(root/'docs').glob('*.html'),*(root/'notebooks').glob('*.ipynb')]
    moved_to=dict(moves)
    for source in documents:
        if not source.is_file():continue
        relative=source.relative_to(root).as_posix()
        if moved_to.get(relative,'').startswith('archive/'):continue
        within(root,relative)
        raw=source.read_bytes();text=raw.decode('utf-8-sig');updated=text
        for old,new in RENAMES.items():updated=updated.replace(Path(old).name,Path(new).name)
        if updated!=text:changes[relative]=updated.encode('utf-8')
    settings=root/'.vscode/settings.json'
    if settings.exists():
        within(root,'.vscode/settings.json')
        value=json.loads(settings.read_text(encoding='utf-8-sig'))
        excluded=value.setdefault('files.exclude',{})
        for key in ['**/__pycache__','**/.ipynb_checkpoints','.venv_ufm','.vscode','tmp']:
            excluded[key]=True
        content=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
        if settings.read_bytes()!=content:changes['.vscode/settings.json']=content
    return moves,changes


def organize(root,apply=False):
    root=root.resolve();moves,changes=plan(root)
    result=dict(moves=moves,updated_documents=list(changes))
    if not apply or not (moves or changes):return result
    protected={p.relative_to(root).as_posix():sha(p) for folder in ['src','tests'] for p in (root/folder).glob('*.py')}
    # These configs/markers are immutable after creation; progress/queue heartbeats
    # are intentionally excluded because their writers remain active.
    for p in (root/'runs').glob('*/config.json'):protected[p.relative_to(root).as_posix()]=sha(p)
    backup=root/'archive/layout-backup.zip'
    if backup.exists():raise FileExistsError('Existing layout backup; inspect before another migration.')
    originals=set(changes)|{old for old,new in moves if not new.startswith('archive/')}
    backup.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(backup,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for old in sorted(originals):z.write(root/old,old)
    with zipfile.ZipFile(backup) as z:
        if z.testzip() is not None:raise ValueError('Backup failed CRC validation')
        for old in originals:
            if z.read(old)!=(root/old).read_bytes():raise ValueError('Backup mismatch')
    manifest=dict(time_utc=datetime.now(timezone.utc).isoformat(),phase='prepared',
                  moves=moves,updated_documents=list(changes),protected_sha256=protected,
                  backup='archive/layout-backup.zip',completed_moves=[])
    marker=root/'archive/layout.json'
    def save():marker.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    for old,content in changes.items():(root/old).write_bytes(content)
    for old,new in moves:
        source,target=within(root,old),within(root,new)
        target.parent.mkdir(parents=True,exist_ok=True)
        source.rename(target)
        manifest['completed_moves'].append([old,new]);save()
    for name,expected in protected.items():
        if sha(root/name)!=expected:raise ValueError('Protected file changed: '+name)
    manifest['phase']='complete';manifest['protected_files_unchanged']=len(protected);save()
    result['backup']=manifest['backup'];result['protected_files_unchanged']=len(protected)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    print(json.dumps(organize(args.root,args.apply),ensure_ascii=False,indent=2))
