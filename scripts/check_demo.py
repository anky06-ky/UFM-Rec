"""Acceptance check against the real loopback demo; no evaluation labels."""
from datetime import datetime,timezone
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'http://127.0.0.1:8765'


def request(path,payload=None):
    req = urllib.request.Request(URL+path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as response: return json.load(response)


status = request('/api/status')
assert status['app']=='ufm-rec-demo'
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
    checks.append(dict(case=name,passed=True,request=payload,response=response))
    print(name,response['elapsed_ms'],flush=True)
report = dict(time_utc=datetime.now(timezone.utc).isoformat(),status=status,checks=checks,
              timing_scope='One request per case, server inference only; not a throughput benchmark.')
(ROOT/'reports/demo_acceptance_2026_09_30.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
