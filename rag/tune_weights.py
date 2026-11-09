"""Wider relevance-weight grid on the cached synthetic queries (fit half -> audit half)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rag'))
from retriever import QueryRetriever, z  # noqa: E402

out = ROOT / 'reports/rag_20261109'
rows = [json.loads(l) for l in (out / 'queries.jsonl').open()]
r = QueryRetriever(ufm_run=None)
leaf = np.array([c.split(' > ')[-1] for c in r.facts['category']], dtype=object)
parts = [{k: z(v).astype(np.float32) for k, v in r.relevance(q['query_en'] or q['message'])[1].items()} for q in rows]
half = len(rows) // 2


def metrics(w, idx):
    rec, catp = {10: [], 50: [], 200: []}, []
    for j in idx:
        q, p = rows[j], parts[j]
        rel = w['lex'] * p['lex'] + w['clip_text'] * p['clip_text'] + w['clip_image'] * p['clip_image']
        allowed = r.mask(q['filters']); allowed[q['row']] = True
        s = np.where(allowed, rel, -np.inf)
        rank = int((s > s[q['row']]).sum()) + 1
        for k in rec: rec[k].append(rank <= k)
        pool = np.argpartition(-s, 400)[:400]
        top = pool[np.argsort(-(z(s[pool]) + w['prior'] * z(r.prior[pool])))[:6]]
        catp.append(float(np.mean(leaf[top] == leaf[q['row']])))
    return {f'recall@{k}': round(float(np.mean(v)), 4) for k, v in rec.items()} | {'category_p@6': round(float(np.mean(catp)), 4)}


grid = [dict(lex=l, clip_text=1.0, clip_image=b, prior=c) for l in [0, 0.1, 0.25, 0.5, 1.0] for b in [0, 0.25, 0.5, 1.0] for c in [0, 0.25, 0.5]]
fit = [(m['recall@50'] + m['category_p@6'], w, m) for w in grid for m in [metrics(w, range(half))]]
best = max(fit, key=lambda x: x[0])[1]
audit = range(half, len(rows))
res = dict(selected=best, audit_selected=metrics(best, audit),
           audit_reference={'clip_text_only': metrics(dict(lex=0, clip_text=1, clip_image=0, prior=0), audit),
                            'previous_default': metrics(dict(lex=1, clip_text=0.8, clip_image=0.5, prior=0.25), audit),
                            'tfidf_only': metrics(dict(lex=1, clip_text=0, clip_image=0, prior=0), audit)},
           top_fit=[dict(w=w, m=m) for _, w, m in sorted(fit, key=lambda x: -x[0])[:8]])
(out / 'weights_wide.json').write_text(json.dumps(res, indent=2))
print(json.dumps({k: res[k] for k in ['selected', 'audit_selected', 'audit_reference']}, indent=1))
