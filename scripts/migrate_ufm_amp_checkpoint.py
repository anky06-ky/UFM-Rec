"""One-time migration for the finite UFM checkpoint stopped by AMP overflow."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'runs/ufm_full_v1'
OLD_SHA = '5b2b205b9e7e22e85ed26f9a1a002d3b55aca0949e31406d138858c32a9be348'


def migrate():
    config_path, checkpoint_path = OUTPUT / 'config.json', OUTPUT / 'latest.pt'
    backup_config = OUTPUT / 'config.pre_amp_fix.json'
    backup_checkpoint = OUTPUT / 'latest.pre_amp_fix.pt'
    if (OUTPUT / 'completed.json').exists() or backup_config.exists() or backup_checkpoint.exists():
        raise RuntimeError('Migration already done or run completed.')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config['code_sha256']['train_ufm_recommender.py'] != OLD_SHA or config['amp'] is not True:
        raise ValueError('Unexpected training configuration; refusing migration.')
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if state['config'] != config or state['global_step'] != 6000 or state['next_batch'] != 6000:
        raise ValueError('Unexpected checkpoint/config/cursor; refusing migration.')
    if not all(torch.isfinite(value).all() for value in state['model'].values()):
        raise ValueError('Checkpoint has nonfinite model weights.')
    for group in state['optimizer']['state'].values():
        if not all(torch.isfinite(value).all() for value in group.values() if torch.is_tensor(value)):
            raise ValueError('Checkpoint has nonfinite optimizer state.')
    new_sha = hashlib.sha256((ROOT / 'src/train_ufm_recommender.py').read_bytes()).hexdigest()
    if new_sha == OLD_SHA:
        raise ValueError('AMP fix not installed yet.')
    migrated = json.loads(json.dumps(config))
    migrated['code_sha256']['train_ufm_recommender.py'] = new_sha
    state['config'] = migrated
    shutil.copyfile(config_path, backup_config)
    shutil.copyfile(checkpoint_path, backup_checkpoint)
    temporary = OUTPUT / 'latest.amp_fix.tmp'
    try:
        torch.save(state, temporary)
        temporary.replace(checkpoint_path)
        temporary = OUTPUT / 'config.amp_fix.tmp'
        temporary.write_text(json.dumps(migrated, indent=2) + '\n', encoding='utf-8')
        temporary.replace(config_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Migrated checkpoint step 6000: {OLD_SHA} -> {new_sha}; originals backed up.', flush=True)


if __name__ == '__main__':
    migrate()
