"""Acceptance check against the real loopback demo; no evaluation labels."""
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--url', default='http://127.0.0.1:8765')
p.add_argument('--backend', choices=('content','ufm'), default='content')
p.add_argument('--output', type=Path, default=ROOT/'reports/demo_acceptance_latest.json')
args = p.parse_args()
URL = args.url.rstrip('/')
if args.output.exists():
    raise FileExistsError('Use a new acceptance report path.')


def request(path,payload=None):
    req = urllib.request.Request(URL+path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=300) as response: return json.load(response)


status = request('/api/status')
assert status['app']=='ufm-rec-demo'
assert status['backend'] == ('UFM Rec' if args.backend=='ufm' else 'TF-IDF content baseline')
results = request('/api/search?q=LEGO')
assert results
history = [results[0]['asin']]
checks = []
for name,selected,regime,k in [('new_user',[],'all',5),('personalized',history,'all',10),
                             ('zero_shot',history,'zero_shot',5),('warm',history,'warm',5)]:
    payload = dict(history=selected,regime=regime,k=k)
    response = request('/api/recommend',payload)
    rows = response['results']
    assert len(rows)==k and len({r['asin'] for r in rows})==k
    assert not set(selected)&{r['asin'] for r in rows}
    if regime!='all': assert all(r['regime']==regime for r in rows)
    if args.backend=='ufm':
        assert all(len(r['weights'])==2 and len(r['uncertainty'])==2 for r in rows)
    checks.append(dict(case=name,passed=True,request=payload,response=response))
    print(name,response['elapsed_ms'],flush=True)
report = dict(time_utc=datetime.now(timezone.utc).isoformat(),status=status,checks=checks,
              timing_scope='One request per case, server inference only; not a throughput benchmark.')
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
