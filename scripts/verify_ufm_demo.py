"""Verify actual loopback UFM requests and save evidence, without test labels."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def verify(url, output, readiness_seconds=900):
    if output.exists():
        raise FileExistsError(output)
    deadline = time.monotonic() + readiness_seconds
    while True:
        try:
            with urllib.request.urlopen(url + '/api/status', timeout=5) as response:
                status = json.load(response)
            break
        except (urllib.error.URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(2)
    assert status['backend'] == 'UFM Rec', status
    assert status['model']['checkpoint_sha256']
    assert not status['data']['counts_simulated'], status
    with urllib.request.urlopen(url + '/api/examples', timeout=10) as response:
        examples = json.load(response)
    history = [examples[0]['asin']]
    cases = []
    for name, past, regime in [('no_history', [], 'all'), ('history', history, 'all')] + [
            (regime, history, regime) for regime in ['zero_shot', 'extreme_cold', 'cold', 'warm']]:
        value = dict(history=past, k=5, regime=regime)
        request = urllib.request.Request(url + '/api/recommend', json.dumps(value).encode(),
                                         {'Content-Type': 'application/json'})
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.load(response)
        assert result['backend'] == 'UFM Rec'
        assert result['model'] == status['model']
        assert len(result['results']) == 5
        for item in result['results']:
            assert item['asin'] not in past
            assert regime == 'all' or item['regime'] == regime
            assert len(item['weights']) == 2 and len(item['uncertainty']) == 2
        cases.append(dict(name=name, request=value, response=result,
                          wall_seconds=time.perf_counter()-start))
        print('DEMO_CASE_PASS', name, flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),
            phase='complete' if len(cases)==6 else 'running', status=status, cases=cases,
            limitations=['API correctness evidence; not recommendation-quality evidence.',
                         'Sequential CPU requests; not a concurrent latency benchmark.']),
            ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8770')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--readiness-seconds', type=int, default=900)
    args = parser.parse_args()
    verify(args.url.rstrip('/'), args.output, args.readiness_seconds)
