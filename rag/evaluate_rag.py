"""Offline evaluation of the RAG layer.

1. Query set: N target products sampled (seed 42) from validation-period purchases with a category
   and feature bullets. The local LLM writes a Vietnamese shopper request from category + bullets
   only (no title, no brand), so lexical overlap with the title is not given away.
2. Retrieval: the pipeline's intent step rewrites each request; we measure, over the full 767k catalog,
   exact-target Recall@{10,50,200} and category precision@6 (share of shown products whose leaf
   category equals the target's). Relevance weights are chosen on the fit half (by sample order)
   and reported on the audit half.
3. Answers (subset): citation validity, share of answers citing >=1 product, and price grounding
   (every "$x" in the answer must equal a price in the context).
Test interactions are never read.
"""
import argparse, csv, gzip, json, random, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rag'))
from retriever import QueryRetriever, WEIGHTS, z  # noqa: E402
from llm import LocalLLM  # noqa: E402
from pipeline import RAGPipeline  # noqa: E402

GEN_SYSTEM = """You write one realistic message that a Vietnamese parent or shopper might type into a toy store chat when looking for a product like the one described.
Write in Vietnamese only (no Chinese, no English sentences), 8-25 words, short and casual like a real chat message, describe the need (who it is for, what kind of toy, key features). Do not mention brand names, model numbers or the exact product name. Output only the message."""


def sample_targets(r, n, seed=42):
    data = r.data
    seen, cands = set(), []
    with gzip.open(data / 'interactions_validation.csv.gz', 'rt', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            i = r.catalog.lookup.get(row['parent_asin'])
            if i is None or i in seen:
                continue
            seen.add(i)
            if r.facts['category'][i] and len(r.facts['bullets'][i]) > 60:
                cands.append(i)
    random.Random(seed).shuffle(cands)
    return cands[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--answers', type=int, default=80)
    ap.add_argument('--out', type=Path, default=ROOT / 'reports/rag_20261109')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    llm = LocalLLM()
    r = QueryRetriever(ufm_run=None)
    pipe = RAGPipeline(r, llm)
    cache = args.out / 'queries.jsonl'
    rows = [json.loads(l) for l in cache.open()] if cache.exists() else []
    done = {q['row'] for q in rows}
    targets = sample_targets(r, args.n)
    with cache.open('a') as f:
        for i in targets:
            if i in done:
                continue
            desc = f"Category: {r.facts['category'][i]}\nFeatures: {r.facts['bullets'][i][:500]}"
            age = r.facts['age_min'][i]
            if np.isfinite(age):
                desc += f'\nRecommended age: {age:g}+'
            for attempt in range(4):  # small models sometimes drift into Chinese; retry those
                msg, _ = llm.generate([{'role': 'system', 'content': GEN_SYSTEM}, {'role': 'user', 'content': desc}], 80, temperature=0.4)
                msg = msg.strip().strip('"')
                if not re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', msg):
                    break
            else:
                continue
            intent, _ = pipe.understand(msg, {})
            q = dict(row=int(i), asin=r.catalog.asins[i], category=r.facts['category'][i], message=msg,
                     query_en=intent['query_en'], filters=intent['filters'], exclude=intent['exclude'])
            f.write(json.dumps(q, ensure_ascii=False) + '\n'); f.flush()
            rows.append(q)
            if len(rows) % 25 == 0:
                print(f'{len(rows)} queries, {time.monotonic() - t0:.0f}s', flush=True)
    rows = [q for q in rows if q['row'] in set(targets)]
    print('queries', len(rows), flush=True)

    # Retrieval metrics; relevance parts cached per query.
    leaf = np.array([c.split(' > ')[-1] for c in r.facts['category']], dtype=object)
    parts = []
    for q in rows:
        _, p = r.relevance(q['query_en'] or q['message'])
        parts.append({k: z(v).astype(np.float32) for k, v in p.items()})
    half = len(rows) // 2

    def metrics(w, idx, use_filters=True):
        rec = {10: [], 50: [], 200: []}
        catp = []
        for j in idx:
            q, p = rows[j], parts[j]
            rel = p['lex'] + w['clip_text'] * p['clip_text'] + w['clip_image'] * p['clip_image']
            allowed = r.mask(q['filters']) if use_filters else np.ones(len(rel), bool)
            allowed[q['row']] = True  # never let a mis-parsed filter hide the target itself from recall
            score = np.where(allowed, rel, -np.inf)
            rank = int((score > score[q['row']]).sum()) + 1
            for k in rec:
                rec[k].append(rank <= k)
            pool = np.argpartition(-score, 400)[:400]
            final = z(score[pool]) + w['prior'] * z(r.prior[pool])
            top = pool[np.argsort(-final)[:6]]
            catp.append(float(np.mean(leaf[top] == leaf[q['row']])))
        return {f'recall@{k}': round(float(np.mean(v)), 4) for k, v in rec.items()} | {'category_p@6': round(float(np.mean(catp)), 4)}

    grid = [dict(clip_text=a, clip_image=b, prior=c) for a in [0, 0.4, 0.8, 1.2] for b in [0, 0.25, 0.5, 1.0] for c in [0, 0.25, 0.5]]
    fit = list(range(half))
    scored = []
    for w in grid:
        m = metrics(w, fit)
        scored.append((m['recall@50'] + m['category_p@6'], w, m))
    best = max(scored, key=lambda x: x[0])[1]
    audit = list(range(half, len(rows)))
    variants = {
        'tfidf_only': dict(clip_text=0, clip_image=0, prior=0),
        'clip_only_text': dict(clip_text=1e6, clip_image=0, prior=0),
        'served_default': {k: WEIGHTS[k] for k in ['clip_text', 'clip_image', 'prior']},
        'selected_on_fit': best,
    }
    retrieval = {k: dict(weights=w, audit=metrics(w, audit)) for k, w in variants.items()}
    raw = [dict(q, query_en=q['message']) for q in rows]  # Vietnamese text straight into retrieval, no LLM rewrite
    saved, rows[:] = rows[:], raw
    parts_saved, parts[:] = parts[:], []
    for q in rows:
        _, p = r.relevance(q['query_en'])
        parts.append({k: z(v).astype(np.float32) for k, v in p.items()})
    retrieval['no_llm_rewrite'] = dict(weights=best, audit=metrics(best, audit))
    rows[:], parts[:] = saved, parts_saved
    print(json.dumps(retrieval, indent=1), flush=True)

    # Answer grounding on a subset of audit queries.
    ans = []
    for j in audit[:args.answers]:
        q = rows[j]
        products, _ = r.search(q['query_en'] or q['message'], q['filters'], (), q['exclude'], k=6, weights=WEIGHTS | best)
        if not products:
            continue
        text, cited, invalid, st = pipe.answer(q['message'], products, q['filters'])
        prices = {round(p['price'], 2) for p in products if p['price'] is not None}
        said = [float(x) for x in re.findall(r'\$\s*(\d+(?:\.\d+)?)', text)] + \
               [float(x) for x in re.findall(r'(\d+(?:\.\d+)?)\s*(?:\$|USD|đô)', text)]
        bad = [x for x in said if all(abs(x - p) > 0.011 for p in prices)]
        ans.append(dict(row=q['row'], cited=cited, invalid=invalid, prices_said=said, prices_ungrounded=bad,
                        json_failed=bool(st.get('json_failed')), seconds=st['seconds'], answer=text))
    answers = dict(n=len(ans), cite_rate=round(np.mean([bool(a['cited']) for a in ans]), 4),
                   invalid_citation_rate=round(np.mean([bool(a['invalid']) for a in ans]), 4),
                   answers_with_price=int(sum(bool(a['prices_said']) for a in ans)),
                   ungrounded_price_rate=round(sum(len(a['prices_ungrounded']) for a in ans) / max(1, sum(len(a['prices_said']) for a in ans)), 4),
                   json_failed_rate=round(np.mean([a['json_failed'] for a in ans]), 4),
                   mean_seconds=round(float(np.mean([a['seconds'] for a in ans])), 2))
    result = dict(protocol=__doc__, n_queries=len(rows), fit=len(fit), audit=len(audit), llm=llm.name, llm_device=llm.device,
                  selected_weights=best, retrieval=retrieval, answers=answers,
                  grid_fit=[dict(w=w, m=m) for _, w, m in scored], runtime_seconds=round(time.monotonic() - t0))
    (args.out / 'rag_eval.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    with (args.out / 'answers.jsonl').open('w') as f:
        for a in ans:
            f.write(json.dumps(a, ensure_ascii=False) + '\n')
    print(json.dumps(dict(selected=best, answers=answers), indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
