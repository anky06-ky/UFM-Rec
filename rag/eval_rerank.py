"""Stage-2 ablation on the cached synthetic queries: does a cross-encoder re-ranker help?
Pool = top-100 by the served relevance blend (filters applied). Parameters on fit half, numbers on audit half."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rag'))
from retriever import QueryRetriever, WEIGHTS, z  # noqa: E402
from advanced import Reranker, doc_text  # noqa: E402

out = ROOT / 'reports/rag_20261109'
rows = [json.loads(l) for l in (out / 'queries.jsonl').open()]
r = QueryRetriever(ufm_run=None)
leaf = np.array([c.split(' > ')[-1] for c in r.facts['category']], dtype=object)
POOL = 100
pools = []
t0 = time.monotonic()
for q in rows:
    rel, _ = r.relevance(q['query_en'] or q['message'])
    allowed = r.mask(q['filters']); allowed[q['row']] = True
    s = np.where(allowed, rel, -np.inf)
    pool = np.argpartition(-s, POOL)[:POOL]
    pool = pool[np.argsort(-s[pool])]
    pools.append(dict(pool=pool, rel=s[pool], prior=r.prior[pool], in_pool=bool(q['row'] in set(pool.tolist()))))
print('pools', time.monotonic() - t0, 'target in top100:', np.mean([p['in_pool'] for p in pools]), flush=True)
ce = {}
for name in ['minilm', 'bge']:
    rk = Reranker(name)
    t1 = time.monotonic()
    ce[name] = [rk.score(q['query_en'] or q['message'], [doc_text(r.facts, r.catalog.titles[i], i) for i in p['pool']])
                for q, p in zip(rows, pools)]
    print(name, rk.device, 'sec/query', round((time.monotonic() - t1) / len(rows), 3), flush=True)
    del rk


def metrics(idx, a, b, name):
    hit10, catp, mrr = [], [], []
    for j in idx:
        p, q = pools[j], rows[j]
        f = z(p['rel']) + b * z(p['prior'])
        if name:
            f = f + a * z(ce[name][j])
        order = p['pool'][np.argsort(-f)]
        pos = np.flatnonzero(order == q['row'])
        hit10.append(bool(len(pos) and pos[0] < 10))
        mrr.append(1 / (pos[0] + 1) if len(pos) else 0)
        catp.append(float(np.mean(leaf[order[:6]] == leaf[q['row']])))
    return dict(recall10=round(float(np.mean(hit10)), 4), mrr100=round(float(np.mean(mrr)), 4), category_p6=round(float(np.mean(catp)), 4))


half = len(rows) // 2
fit, audit = range(half), range(half, len(rows))
res = dict(baseline=metrics(audit, 0, WEIGHTS['prior'], None))
for name in ce:
    grid = [(a, b) for a in [0.5, 1, 2, 4, 1000] for b in [0, 0.25, 0.5]]
    best = max(grid, key=lambda ab: (lambda m: m['category_p6'] + m['mrr100'])(metrics(fit, *ab, name)))
    res[name] = dict(weights=dict(ce=best[0], prior=best[1]), audit=metrics(audit, *best, name))
(out / 'rerank_ablation.json').write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=1))
