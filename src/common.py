import hashlib
import json
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)

import numpy as np

def push_unseen_last(scores, seen):
    floor = np.where(seen, scores, np.inf).min(1, keepdims=True) - 1.0
    floor[~seen.any(1)] = 0.0
    return np.where(seen, scores, floor)

