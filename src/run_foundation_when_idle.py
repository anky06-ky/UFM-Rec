"""Run CLIP smoke + full extraction AFTER the existing baseline finishes.

Does not kill/restart training, fit a recommender, or open evaluation samples.
Use an external flock to prevent duplicate queue runners. All stages use new
output directories or resume exactly matching extraction progress.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/toys_games_full_temporal"


def active_baseline_pids(proc_root=Path('/proc')):
    result = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            argv = (directory / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        expected = {b'src/train_gpu_recommender.py', str(ROOT / 'src/train_gpu_recommender.py').encode()}
        if any(x in expected for x in argv):
            result.append(int(directory.name))
    return result


def baseline_finished(root=ROOT):
    marker = root / 'runs/content_transformer_v1/completed.json'
    if not marker.exists():
        return False
    state = json.loads(marker.read_text(encoding='utf-8'))
    if state.get('smoke_run') is not False or not (root / 'runs/content_transformer_v1/best.pt').is_file():
        raise ValueError('Baseline completion must describe a full, checkpointed run.')
    return True


def gpu_idle():
    status = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()[0]
    free, utilization = (int(x.strip()) for x in status.split(','))
    return free >= 8192 and utilization <= 20, {'free_MiB': free, 'utilization_pct': utilization}


def atomic_status(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def extraction_command(output, limit, batch):
    command = [sys.executable, '-u', str(ROOT / 'src/extract_foundation_features.py'),
               '--device', 'cuda', '--limit', str(limit), '--batch-size', str(batch),
               '--catalog', str(DATA / 'foundation_catalog_v1'), '--output', str(output),
               '--model-cache', str(ROOT / 'tmp/hf_ufm_models')]
    if (output / 'progress.json').exists() and not (output / 'complete.json').exists():
        command.append('--resume')
    return command


def verify_smoke(output):
    import numpy as np
    from prepare_ufm_catalog import sha256
    marker = json.loads((output / 'complete.json').read_text(encoding='utf-8'))
    if marker['rows'] != 128 or marker['device'] != 'cuda' or not marker['encoder_frozen']:
        raise ValueError('Expected a completed 128-item frozen-encoder CUDA smoke cache.')
    if marker['has_text'] < 120 or marker['has_image'] < 116:
        raise ValueError('Smoke coverage is too low; diagnose downloads before full extraction.')
    for name in ('text.npy', 'image.npy', 'modalities.npy'):
        if sha256(output / name) != marker['features_sha256'][name]:
            raise ValueError(f'Smoke checksum mismatch: {name}')
    flags = np.load(output / 'modalities.npy')
    if flags.shape != (129, 2) or flags[0].any():
        raise ValueError('Invalid smoke modality mask/padding.')
    for column, name in enumerate(('text.npy', 'image.npy')):
        matrix = np.load(output / name).astype(np.float32)
        if matrix.shape != (129, 512) or not np.isfinite(matrix).all() or matrix[0].any():
            raise ValueError('Invalid feature shape/finite values/padding.')
        np.testing.assert_allclose(np.linalg.norm(matrix[flags[:, column]], axis=-1), 1, atol=0.002)
        if np.any(matrix[~flags[:, column]]):
            raise ValueError('Unavailable modality must be all zero.')
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', type=Path, default=ROOT / 'runs/foundation_queue_v1.json')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--max-wait-hours', type=float, default=24)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60 or args.max_wait_hours <= 0:
        parser.error('Poll 1-60 seconds; positive wait limit required.')
    if not (DATA / 'foundation_catalog_v1/catalog_report.json').exists():
        raise FileNotFoundError('Build/verify the foundation catalog first.')
    started = time.monotonic()
    idle_checks = 0

    def report(stage, **values):
        state = {'stage': stage, 'time_utc': datetime.now(timezone.utc).isoformat(), **values}
        atomic_status(args.status, state)
        print(json.dumps(state), flush=True)

    try:
        while True:
            pids = active_baseline_pids()
            finished = baseline_finished()
            ready, gpu = gpu_idle()
            idle_checks = idle_checks + 1 if finished and not pids and ready else 0
            if idle_checks >= 2:
                break
            report('waiting_for_baseline_and_idle_gpu', baseline_complete=finished,
                   baseline_pids=pids, gpu=gpu, consecutive_idle_checks=idle_checks)
            if time.monotonic() - started >= args.max_wait_hours * 3600:
                raise TimeoutError('Baseline/GPU not ready; no additional GPU job started.')
            time.sleep(args.poll_seconds)
        smoke = ROOT / 'runs/foundation_smoke_128_v1'
        full = DATA / 'foundation_clip_b32_v1'
        report('smoke_running', output=str(smoke))
        if not (smoke / 'complete.json').exists():
            subprocess.run(extraction_command(smoke, 128, 32), cwd=ROOT, check=True)
        verified = verify_smoke(smoke)
        report('smoke_passed', text=verified['has_text'], image=verified['has_image'])
        # Recheck immediately; never stop another job or ignore a busy GPU.
        if active_baseline_pids() or not gpu_idle()[0]:
            raise RuntimeError('GPU became busy; smoke kept, full extraction not started.')
        report('full_extraction_running', output=str(full), expected_rows=767045)
        subprocess.run(extraction_command(full, 0, 64), cwd=ROOT, check=True)
        marker = json.loads((full / 'complete.json').read_text(encoding='utf-8'))
        if marker['rows'] != 767045 or marker['limit'] != 0 or marker['device'] != 'cuda':
            raise ValueError('Full cache marker does not match the intended dataset.')
        report('extraction_complete', output=str(full), text=marker['has_text'],
               image=marker['has_image'], recommender_trained=False, test_evaluated=False)
    except Exception as error:
        report('stopped_with_error', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
