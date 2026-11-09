"""Compare LLMs for the two RAG steps on the held-out synthetic queries (products disjoint from training).

Intent step (all queries): JSON validity and retrieval quality of the produced plan through the served
v2 retrieval (multi-query + HyDE averaged, cross-encoder, filters): exact-target recall@10 / MRR in
the top-100 and leaf-category precision of the shown 6.
Answer step (first --answers queries): identical product contexts for every model (built once from the
1.5B plans); JSON validity, >=2 valid picks, Vietnamese reasons, grounding (numbers in a reason must
appear in that product's context), agreement with the teacher's picks, latency per answer (unbatched).
"""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rag'))
from retriever import QueryRetriever, z  # noqa: E402
from llm import LocalLLM, generate_batch, parse_json  # noqa: E402
from pipeline import INTENT_SYSTEM_V2, ANSWER_SYSTEM, parse_constraints  # noqa: E402
from pipeline_v2 import AdvancedRAG, _num_tokens  # noqa: E402
from advanced import doc_text  # noqa: E402

VI = re.compile(r'[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹáàãéèíìóòõúùý]', re.I)
REP = ROOT / 'reports/rag_20261109'


def plan_texts(o, fallback):
    o = o if isinstance(o, dict) else {}
    main = str(o.get('query_en') or fallback).strip()
    alts = [a for a in (o.get('alt_queries') or []) if isinstance(a, str) and a.strip()][:2]
    hyde = o.get('hyde') if isinstance(o.get('hyde'), str) else ''
    return main, [main] + alts + ([hyde] if hyde.strip() else []), [str(x) for x in (o.get('exclude') or [])][:5]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models', nargs='+', default=['Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-0.5B-Instruct@runs/rag_student_0.5b/adapter.pt',
                                                    'Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-3B-Instruct'])
    ap.add_argument('--teacher', default='Qwen/Qwen2.5-3B-Instruct')
    ap.add_argument('--answers', type=int, default=80)
    ap.add_argument('--out', type=Path, default=REP / 'distill_eval.json')
    args = ap.parse_args()
    rows = [json.loads(l) for l in (REP / 'queries.jsonl').open()]
    r = QueryRetriever(ufm_run=None)
    leaf = np.array([c.split(' > ')[-1] for c in r.facts['category']], dtype=object)

    # Fixed answer contexts from the 1.5B plans (eval_expansion cache).
    plans15 = {json.loads(l)['row']: json.loads(l) for l in (REP / 'intents_v2_Qwen2.5-1.5B-Instruct.jsonl').open()}
    rag = AdvancedRAG(r, llm=None)
    ctx_cases = []
    for q in rows[len(rows) // 2:][:args.answers]:
        main, texts, excl = plan_texts(plans15[q['row']]['obj'], q['query_en'] or q['message'])
        filters = parse_constraints(q['message'])
        res = rag.retrieve(dict(query_en=main, alt_queries=texts[1:3], hyde=texts[3] if len(texts) > 3 else '', exclude=excl), filters, ())
        if res is None or not res['chosen']:
            continue
        products = [dict(r.catalog.item(int(res['pool'][i])), **r.fact(int(res['pool'][i]))) for i in res['chosen']]
        ctx = rag.context(products, main)
        notes = []
        if filters.get('max_price') is not None:
            notes.append(f"budget up to ${filters['max_price']:g}")
        if filters.get('age_years') is not None:
            notes.append(f"child age {filters['age_years']:g}")
        extra = f"\nAll products above already satisfy: {', '.join(notes)}." if notes else ''
        ctx_cases.append(dict(row=q['row'], n=len(products), ctx=ctx, messages=[{'role': 'system', 'content': ANSWER_SYSTEM},
                              {'role': 'user', 'content': f"CONTEXT:\n{ctx}{extra}\n\nUSER REQUEST: {q['message']}"}]))
    print('answer contexts', len(ctx_cases), flush=True)

    results, picks_by_model = {}, {}
    for spec in args.models:
        name, _, adapter = spec.partition('@')
        llm = LocalLLM(name, 'cuda', adapter=str(ROOT / adapter) if adapter else None)
        rag.llm = llm
        t0 = time.monotonic()
        outs = generate_batch(llm, [[{'role': 'system', 'content': INTENT_SYSTEM_V2}, {'role': 'user', 'content': q['message']}] for q in rows], 160)
        objs = [parse_json(o) for o in outs]
        hit10, mrr, catp = [], [], []
        for q, o in zip(rows, objs):
            main, texts, excl = plan_texts(o, q['message'])
            rel = r.relevance_many(texts).mean(0)
            allowed = r.mask(parse_constraints(q['message'])); allowed[q['row']] = True
            s = np.where(allowed, rel, -np.inf)
            pool = np.argpartition(-s, 100)[:100]
            ce = rag.rk.score(main, [doc_text(r.facts, r.catalog.titles[i], i) for i in pool])
            order = pool[np.argsort(-(z(s[pool]) + 0.5 * z(r.prior[pool]) + 4 * z(ce)))]
            pos = np.flatnonzero(order == q['row'])
            hit10.append(bool(len(pos) and pos[0] < 10)); mrr.append(1 / (pos[0] + 1) if len(pos) else 0)
            catp.append(float(np.mean(leaf[order[:6]] == leaf[q['row']])))
        intent = dict(json_ok=round(float(np.mean([isinstance(o, dict) and 'query_en' in o for o in objs])), 4),
                      has_hyde=round(float(np.mean([isinstance(o, dict) and bool(o.get('hyde')) for o in objs])), 4),
                      recall10=round(float(np.mean(hit10)), 4), mrr100=round(float(np.mean(mrr)), 4), category_p6=round(float(np.mean(catp)), 4))
        # answers: unbatched for latency
        ans, lat = [], []
        for c in ctx_cases:
            text, st = llm.generate(c['messages'], 260)
            lat.append(st['seconds'])
            o = parse_json(text)
            blocks = {n: b for n, b in enumerate(re.split(r'\n(?=\[\d+\] )', c['ctx']), 1)}
            ok = isinstance(o, dict) and isinstance(o.get('picks'), list)
            picks = [p for p in (o['picks'] if ok else []) if isinstance(p, dict) and isinstance(p.get('n'), int)]
            valid = [p for p in picks if 1 <= p['n'] <= c['n']]
            whys = [str(p.get('why') or '') for p in valid]
            ungrounded = sum(bool(_num_tokens(w) - _num_tokens(blocks[p['n']])) for w, p in zip(whys, valid))
            ans.append(dict(row=c['row'], json_ok=ok, n_valid=len(valid), invalid=len(picks) - len(valid),
                            vi=bool(whys) and all(VI.search(w) for w in whys), ungrounded=ungrounded, picks=[p['n'] for p in valid]))
        picks_by_model[spec] = {a['row']: a['picks'] for a in ans}
        answer = dict(json_ok=round(float(np.mean([a['json_ok'] for a in ans])), 4),
                      two_plus_valid=round(float(np.mean([a['n_valid'] >= 2 for a in ans])), 4),
                      invalid_pick_rate=round(sum(a['invalid'] for a in ans) / max(1, sum(a['invalid'] + a['n_valid'] for a in ans)), 4),
                      vietnamese=round(float(np.mean([a['vi'] for a in ans])), 4),
                      ungrounded_reason_rate=round(sum(a['ungrounded'] for a in ans) / max(1, sum(a['n_valid'] for a in ans)), 4),
                      seconds_p50=round(float(np.median(lat)), 2), seconds_mean=round(float(np.mean(lat)), 2))
        results[spec] = dict(intent=intent, answer=answer, minutes=round((time.monotonic() - t0) / 60, 1),
                             vram_gib=round(llm.torch.cuda.max_memory_allocated() / 2**30, 2))
        print(spec, json.dumps(results[spec]), flush=True)
        del llm, rag.llm
        import torch; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    if args.teacher in picks_by_model:
        tp = picks_by_model[args.teacher]
        for spec in results:
            agree = [len(set(p[:3]) & set(tp.get(row, [])[:3])) / max(1, len(set(tp.get(row, [])[:3]) | set(p[:3])))
                     for row, p in picks_by_model[spec].items() if tp.get(row)]
            results[spec]['answer']['jaccard_top3_vs_teacher'] = round(float(np.mean(agree)), 4) if agree else None
    args.out.write_text(json.dumps(dict(protocol=__doc__, n_intent=len(rows), n_answer=len(ctx_cases), results=results), indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
