"""Build the distillation set: teacher LLM outputs, filtered by automatic checks, for the two LLM steps.

Products used here are disjoint from the evaluation targets (reports/rag_20261109/queries.jsonl).
1. messages: the teacher writes a Vietnamese shopper request from category + feature bullets of a
   sampled product (no title/brand); some get a price/age phrase; some become follow-up turns.
2. intent labels (retrieval-feedback selection): the teacher proposes K plans (greedy + sampled);
   each plan is executed by the retriever and scored by how well it retrieves the source product's
   leaf category in the top 6 and the source product itself. The best valid plan is kept, so the
   student learns from the teacher's best behaviour rather than its average one.
3. answer labels: the v2 retrieval (cross-encoder, MMR) runs with the kept plan; the teacher writes the
   picks JSON; kept only if JSON is valid, >=2 existing picks, reasons in Vietnamese and every number
   in a reason appears in that product's context (grounding).
Output: data/processed/toys_games_full_temporal/rag_distill_v1/{messages,intent_sft,answer_sft}.jsonl
"""
import argparse, json, random, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rag'))
from retriever import QueryRetriever, z  # noqa: E402
from llm import LocalLLM, generate_batch, parse_json  # noqa: E402
from pipeline import INTENT_SYSTEM_V2, ANSWER_SYSTEM, parse_constraints  # noqa: E402
from pipeline_v2 import AdvancedRAG, _num_tokens  # noqa: E402
from evaluate_rag import GEN_SYSTEM  # noqa: E402

OUT = ROOT / 'data/processed/toys_games_full_temporal/rag_distill_v1'
CJK = re.compile(r'[぀-ヿ一-鿿]')
VI = re.compile(r'[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹáàãéèíìóòõúùýđ]', re.I)
FOLLOW = ['có loại nào rẻ hơn không?', 'có loại điều khiển từ xa không?', 'loại không cần dùng pin', 'cho bé gái thì sao?',
          'có loại bằng gỗ không?', 'loại nào cho người lớn chơi được?', 'có bộ nào nhiều món hơn không?', 'loại nhỏ gọn mang đi du lịch']
NEW_TOPIC = ['còn đồ chơi nấu ăn cho bé gái thì sao?', 'thôi, tìm giúp mình bộ lego xe cảnh sát', 'cho mình xem board game cho cả nhà',
             'tìm puzzle 500 mảnh cho người lớn', 'đổi sang đồ chơi tắm cho bé 1 tuổi']


def jl(path):
    return [json.loads(l) for l in path.open()] if path.exists() else []


def append(path, rec):
    with path.open('a') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def valid_plan(o):
    return isinstance(o, dict) and isinstance(o.get('query_en'), str) and isinstance(o.get('alt_queries', []), list) \
        and isinstance(o.get('hyde', ''), str) and isinstance(o.get('exclude', []), list) and 'chit_chat' in o


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--teacher', default='Qwen/Qwen2.5-3B-Instruct')
    ap.add_argument('--n', type=int, default=900)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--answers', type=int, default=700)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    global OUT
    OUT = args.out or OUT
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    llm = LocalLLM(args.teacher, 'cuda')
    r = QueryRetriever(ufm_run=None)
    rag = AdvancedRAG(r, llm)
    leaf = np.array([c.split(' > ')[-1] for c in r.facts['category']], dtype=object)
    eval_rows = {q['row'] for q in jl(ROOT / 'reports/rag_20261109/queries.jsonl')}
    rng = random.Random(7)

    # 1. messages
    mpath = OUT / 'messages.jsonl'
    msgs = jl(mpath)
    if len(msgs) < args.n:
        ok = np.flatnonzero(np.fromiter((len(b) > 60 and bool(c) for b, c in zip(r.facts['bullets'], r.facts['category'])), bool, len(r.catalog.asins)))
        pool = [int(i) for i in ok if int(i) not in eval_rows and r.catalog.counts[i] > 0]
        rng.shuffle(pool)
        pool = pool[len(msgs):len(msgs) + 3 * args.n]
        while len(msgs) < args.n and pool:
            part, pool = pool[:args.batch * 4], pool[args.batch * 4:]
            convs = []
            for i in part:
                desc = f"Category: {r.facts['category'][i]}\nFeatures: {r.facts['bullets'][i][:500]}"
                if np.isfinite(r.facts['age_min'][i]):
                    desc += f"\nRecommended age: {r.facts['age_min'][i]:g}+"
                convs.append([{'role': 'system', 'content': GEN_SYSTEM}, {'role': 'user', 'content': desc}])
            for i, m in zip(part, generate_batch(llm, convs, 70, 0.8, args.batch)):
                m = m.strip().strip('"')
                if CJK.search(m) or not VI.search(m) or len(m) < 15:
                    continue
                price = r.facts['price'][i]
                u = rng.random()
                if u < 0.25 and np.isfinite(price):
                    m += rng.choice([f', dưới {int(price * 1.3) + 1}$', f', tầm {int(price * 1.3 * 25 + 1) * 1000 // 1000}k'])
                elif u < 0.45 and np.isfinite(r.facts['age_min'][i]):
                    m += f', cho bé {int(max(r.facts["age_min"][i], 1)) + rng.choice([0, 1, 2])} tuổi'
                rec = dict(row=i, message=m)
                msgs.append(rec); append(mpath, rec)
            print('messages', len(msgs), round(time.monotonic() - t0), 's', flush=True)

    # 2. intent labels with retrieval feedback
    ipath = OUT / 'intent_sft.jsonl'
    done = {(x['row'], x['user']) for x in jl(ipath)}
    todo = []
    for m in msgs:
        todo.append(dict(row=m['row'], user=m['message'], prev=None))
    for m in rng.sample(msgs, len(msgs) // 4):  # follow-up turns, resolved against the teacher's first plan later
        todo.append(dict(row=m['row'], user=rng.choice(FOLLOW + NEW_TOPIC), prev=m['message']))
    todo = [t for t in todo if (t['row'], t['user']) not in done]
    first_plan = {x['row']: x['plan'] for x in jl(ipath) if x.get('prev') is None}
    todo.sort(key=lambda t: t['prev'] is not None)  # first turns before follow-ups
    for s in range(0, len(todo), args.batch):
        part = todo[s:s + args.batch]
        users = []
        for t in part:
            if t['prev'] is None:
                users.append(t['user'])
            else:
                q = (first_plan.get(t['row']) or {}).get('query_en', '')
                users.append(f"PREVIOUS: {json.dumps({'query_en': q}, ensure_ascii=False)}  User: {t['user']}")
        convs = [[{'role': 'system', 'content': INTENT_SYSTEM_V2}, {'role': 'user', 'content': u}] for u in users]
        cands = [generate_batch(llm, convs, 160, 0.0, args.batch)] + \
                [generate_batch(llm, convs, 160, 0.7, args.batch) for _ in range(args.k - 1)]
        for j, (t, u) in enumerate(zip(part, users)):
            plans = [parse_json(c[j]) for c in cands]
            plans = [p for p in plans if valid_plan(p)]
            if not plans:
                continue
            scored = []
            for p in plans:
                texts = [p['query_en']] + [a for a in p.get('alt_queries', []) if isinstance(a, str)][:2] + ([p['hyde']] if p.get('hyde') else [])
                texts = [x for x in texts if x.strip()]
                if not texts:
                    scored.append((-1, p)); continue
                if t['prev'] is not None and t['user'] in NEW_TOPIC:
                    scored.append((0 if p.get('refine') else 1, p)); continue  # new topic: only check the refine flag
                rel = r.relevance_many(texts).mean(0)
                top = np.argpartition(-rel, 6)[:6]
                rank = int((rel > rel[t['row']]).sum()) + 1
                score = float(np.mean(leaf[top] == leaf[t['row']])) + 1.0 / np.log2(1 + rank)
                if t['prev'] is not None:
                    score += 0.5 * bool(p.get('refine'))
                scored.append((score, p))
            best_score, best = max(scored, key=lambda x: x[0])
            target = json.dumps({k: best.get(k, d) for k, d in [('query_en', ''), ('alt_queries', []), ('hyde', ''), ('exclude', []),
                                                                 ('refine', False), ('chit_chat', False)]}, ensure_ascii=False)
            rec = dict(row=t['row'], user=t['user'], prev=t['prev'], prompt_user=u, target=target, score=round(best_score, 4),
                       greedy_score=round(scored[0][0], 4) if plans else None, n_valid=len(plans),
                       messages=[{'role': 'system', 'content': INTENT_SYSTEM_V2}, {'role': 'user', 'content': u}])
            append(ipath, rec)
            if t['prev'] is None:
                first_plan[t['row']] = best
        print('intents', s + len(part), '/', len(todo), round(time.monotonic() - t0), 's', flush=True)

    # 3. answer labels
    apath = OUT / 'answer_sft.jsonl'
    have = {x['row'] for x in jl(apath)}
    intents = [x for x in jl(ipath) if x['prev'] is None and x['row'] not in have][:max(0, args.answers - len(have))]
    kept = 0
    for s in range(0, len(intents), args.batch):
        part, convs, ctxs, prods = intents[s:s + args.batch], [], [], []
        for x in part:
            plan = json.loads(x['target'])
            intent = dict(query_en=plan['query_en'], alt_queries=plan['alt_queries'][:2], hyde=plan['hyde'], exclude=plan['exclude'])
            filters = parse_constraints(x['user'])
            res = rag.retrieve(intent, filters, ())
            if res is None or not res['chosen']:
                convs.append(None); ctxs.append(None); prods.append(None); continue
            products = [dict(r.catalog.item(int(res['pool'][i])), **r.fact(int(res['pool'][i]))) for i in res['chosen']]
            ctx = rag.context(products, intent['query_en'])
            notes = []
            if filters.get('max_price') is not None:
                notes.append(f"budget up to ${filters['max_price']:g}")
            if filters.get('age_years') is not None:
                notes.append(f"child age {filters['age_years']:g}")
            extra = f"\nAll products above already satisfy: {', '.join(notes)}." if notes else ''
            convs.append([{'role': 'system', 'content': ANSWER_SYSTEM}, {'role': 'user', 'content': f"CONTEXT:\n{ctx}{extra}\n\nUSER REQUEST: {x['user']}"}])
            ctxs.append(ctx); prods.append(products)
        live = [i for i, c in enumerate(convs) if c is not None]
        cand = [generate_batch(llm, [convs[i] for i in live], 260, 0.0, args.batch),
                generate_batch(llm, [convs[i] for i in live], 260, 0.7, args.batch)]
        for jj, i in enumerate(live):
            blocks = {n: b for n, b in enumerate(re.split(r'\n(?=\[\d+\] )', ctxs[i]), 1)}
            for c in (cand[0][jj], cand[1][jj]):
                o = parse_json(c)
                if not isinstance(o, dict) or not isinstance(o.get('picks'), list):
                    continue
                picks = [p for p in o['picks'] if isinstance(p, dict) and isinstance(p.get('n'), int) and 1 <= p['n'] <= len(prods[i])]
                if len(picks) < 2 or len(picks) != len(o['picks']) or len({p['n'] for p in picks}) != len(picks):
                    continue
                whys = [str(p.get('why') or '') for p in picks]
                if not all(VI.search(w) for w in whys) or not VI.search(str(o.get('intro') or '')) or any(CJK.search(w) for w in whys):
                    continue
                if any(_num_tokens(w) - _num_tokens(blocks[p['n']]) for w, p in zip(whys, picks)):
                    continue
                target = json.dumps(dict(intro=str(o.get('intro') or ''), picks=[dict(n=p['n'], why=str(p.get('why') or '')) for p in picks],
                                         tip=str(o.get('tip') or '')), ensure_ascii=False)
                append(apath, dict(row=part[i]['row'], user=part[i]['user'], messages=convs[i], target=target))
                kept += 1
                break
        print('answers', s + len(part), '/', len(intents), 'kept', kept, round(time.monotonic() - t0), 's', flush=True)
    summary = dict(messages=len(jl(mpath)), intent_examples=len(jl(ipath)), answer_examples=len(jl(apath)), teacher=args.teacher,
                   seconds=round(time.monotonic() - t0))
    (OUT / 'complete.json').write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == '__main__':
    main()
