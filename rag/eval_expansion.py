"""Query expansion ablation (multi-query, HyDE, RRF) with cross-encoder re-ranking on top.
The served LLM rewrites each cached synthetic request with INTENT_SYSTEM_V2; fit/audit halves as before."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rag'))
from retriever import QueryRetriever, z  # noqa: E402
from advanced import Reranker, doc_text, rrf  # noqa: E402
from llm import LocalLLM  # noqa: E402
from pipeline import INTENT_SYSTEM_V2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--llm-device', default='cpu')
ap.add_argument('--llm-model', default='Qwen/Qwen2.5-1.5B-Instruct')
args = ap.parse_args()
out = ROOT / 'reports/rag_20261109'
rows = [json.loads(l) for l in (out / 'queries.jsonl').open()]
cache = out / f"intents_v2_{args.llm_model.split('/')[-1]}.jsonl"
have = {json.loads(l)['row']: json.loads(l) for l in cache.open()} if cache.exists() else {}
todo = [q for q in rows if q['row'] not in have]
if todo:
    llm = LocalLLM(args.llm_model, args.llm_device)
    t0 = time.monotonic()
    from llm import generate_batch, parse_json
    with cache.open('a') as f:
        for s0 in range(0, len(todo), 8):
            part = todo[s0:s0 + 8]
            t1 = time.monotonic()
            outs = generate_batch(llm, [[{'role': 'system', 'content': INTENT_SYSTEM_V2}, {'role': 'user', 'content': q['message']}] for q in part], 160)
            dt = (time.monotonic() - t1) / len(part)
            for q, raw in zip(part, outs):
                obj = parse_json(raw)
                rec = dict(row=q['row'], obj=obj if isinstance(obj, dict) else None, raw=raw, seconds=round(dt, 2), batched=True)
                f.write(json.dumps(rec, ensure_ascii=False) + '\n'); f.flush()
                have[q['row']] = rec
            print(s0 + len(part), 'intents', round(time.monotonic() - t0), 's', flush=True)
    del llm

r = QueryRetriever(ufm_run=None)
rk = Reranker('minilm')
leaf = np.array([c.split(' > ')[-1] for c in r.facts['category']], dtype=object)


def texts(q):
    o = have[q['row']]['obj'] or {}
    main = (o.get('query_en') or q['query_en'] or q['message']).strip()
    alts = [a for a in (o.get('alt_queries') or []) if isinstance(a, str) and a.strip()][:2]
    hyde = o.get('hyde') if isinstance(o.get('hyde'), str) else ''
    return main, alts, hyde.strip()


cache_rel = {}


def rel(text):
    if text not in cache_rel:
        cache_rel[text] = r.relevance(text)[0].astype(np.float32)
    return cache_rel[text]


VARIANTS = ['v2_main', 'multi_avg', 'multi_hyde_avg', 'multi_hyde_rrf']
pools = {v: [] for v in VARIANTS}
t0 = time.monotonic()
for n, q in enumerate(rows, 1):
    main, alts, hyde = texts(q)
    allowed = r.mask(q['filters']); allowed[q['row']] = True
    sets = {'v2_main': [main], 'multi_avg': [main] + alts, 'multi_hyde_avg': [main] + alts + ([hyde] if hyde else [])}
    allt = sets['multi_hyde_avg']
    for t, v in zip(allt, r.relevance_many(allt)):
        cache_rel[t] = v
    for v, ts in sets.items():
        s = np.mean([rel(t) for t in ts], 0)
        s = np.where(allowed, s, -np.inf)
        pool = np.argpartition(-s, 100)[:100]
        pools[v].append((pool[np.argsort(-s[pool])], s))
    lists = []
    for t in sets['multi_hyde_avg']:
        s = np.where(allowed, rel(t), -np.inf)
        p = np.argpartition(-s, 200)[:200]
        lists.append(p[np.argsort(-s[p])])
    fused = rrf(lists)
    ids = np.array(sorted(fused, key=fused.get, reverse=True)[:100])
    pools['multi_hyde_rrf'].append((ids, np.full(len(r.prior), -np.inf, np.float32)))
    pools['multi_hyde_rrf'][-1][1][ids] = [fused[i] for i in ids]
    cache_rel.clear()
    if n % 50 == 0:
        print(n, 'pools', round(time.monotonic() - t0), flush=True)

ce = {v: [rk.score(texts(q)[0], [doc_text(r.facts, r.catalog.titles[i], i) for i in pools[v][j][0]]) for j, q in enumerate(rows)] for v in VARIANTS}


def metrics(v, idx, use_ce=True):
    hit10, hit100, catp, mrr = [], [], [], []
    for j in idx:
        pool, s = pools[v][j]
        f = z(s[pool]) + 0.5 * z(r.prior[pool]) + (4 * z(ce[v][j]) if use_ce else 0)
        order = pool[np.argsort(-f)]
        pos = np.flatnonzero(order == rows[j]['row'])
        hit10.append(bool(len(pos) and pos[0] < 10)); hit100.append(bool(len(pos)))
        mrr.append(1 / (pos[0] + 1) if len(pos) else 0)
        catp.append(float(np.mean(leaf[order[:6]] == leaf[rows[j]['row']])))
    return dict(recall10=round(float(np.mean(hit10)), 4), recall100=round(float(np.mean(hit100)), 4),
                mrr100=round(float(np.mean(mrr)), 4), category_p6=round(float(np.mean(catp)), 4))


half = len(rows) // 2
res = dict(n=len(rows), json_ok=round(float(np.mean([have[q['row']]['obj'] is not None for q in rows])), 4),
           intent_seconds=round(float(np.mean([have[q['row']]['seconds'] for q in rows])), 2),
           fit={v: metrics(v, range(half)) for v in VARIANTS},
           audit={v: metrics(v, range(half, len(rows))) for v in VARIANTS},
           audit_no_ce={v: metrics(v, range(half, len(rows)), False) for v in VARIANTS})
(out / f"expansion_ablation_{args.llm_model.split('/')[-1]}.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=1))
