"""Advanced RAG pipeline (v2).

understand (LLM: main query + alternative queries + HyDE description, regex constraints)
 -> hybrid recall per query text (TF-IDF + CLIP text/image), averaged -> top-100 under hard filters
 -> cross-encoder re-rank + recent popularity + UFM v2 personalisation -> MMR diversity
 -> corrective step: weak evidence => relax filters (and say so) or ask a clarifying question
 -> contextual compression of product text -> LLM picks + reasons (JSON) -> grounding check
 -> facets (price range, categories) and follow-up suggestions.
Events are yielded so the UI can show products before the answer is written.
"""
import json, re, time
import numpy as np

from retriever import WEIGHTS, z
from advanced import Reranker, doc_text, mmr, compress, rrf
from pipeline import (INTENT_SYSTEM_V2, ANSWER_SYSTEM, parse_constraints, render_answer, citations, short_title)

CE_WEIGHT, PRIOR_WEIGHT, POOL = 4.0, 0.5, 100
WEAK_CE = -2.0          # MiniLM logit: below this the best match is not about the request
USE_ALTS, USE_HYDE = True, True   # multi-query + HyDE fused by RRF: reports/rag_20261109/expansion_ablation_*.json


def _num_tokens(s):
    return set(re.findall(r'\d+(?:[.,]\d+)?', s))


class AdvancedRAG:
    def __init__(self, retriever, llm, reranker=None, k=6):
        self.r, self.llm, self.k = retriever, llm, k
        self.rk = reranker or Reranker('minilm')
        self.retriever = retriever

    # ---------- 1. understanding ----------
    def understand(self, message, state):
        prev = state.get('intent') or {}
        user = message if not prev.get('query_en') else \
            f"PREVIOUS: {json.dumps({'query_en': prev['query_en']}, ensure_ascii=False)}  User: {message}"
        obj, raw, st = self.llm.generate_json([{'role': 'system', 'content': INTENT_SYSTEM_V2}, {'role': 'user', 'content': user}], 160)
        if not isinstance(obj, dict):
            obj = dict(query_en=message, fallback=True)
        clean = lambda s: re.sub(r'\s+', ' ', re.sub(r'\$\s*\d+\S*|\d+\s*\$|\b(under|below)\s+\d+\S*', ' ', str(s or ''), flags=re.I)).strip()
        refine = bool(obj.get('refine')) and bool(prev.get('query_en'))
        intent = dict(query_en=clean(obj.get('query_en')),
                      alt_queries=[clean(a) for a in (obj.get('alt_queries') or []) if isinstance(a, str) and a.strip()][:2],
                      hyde=clean(obj.get('hyde')) if isinstance(obj.get('hyde'), str) else '',
                      exclude=[str(x) for x in obj.get('exclude') or []][:5],
                      refine=refine, chit_chat=bool(obj.get('chit_chat')))
        intent['filters'] = parse_constraints(message, prev.get('filters') if refine else None)
        return intent, st

    # ---------- 2. retrieval + re-ranking ----------
    def retrieve(self, intent, filters, history):
        r = self.r
        texts = [intent['query_en']] + (intent['alt_queries'] if USE_ALTS else []) + ([intent['hyde']] if USE_HYDE and intent['hyde'] else [])
        rels = r.relevance_many(texts)
        allowed = r.mask(filters)
        hist = r.catalog.rows(list(history)[-20:]) if history else []
        allowed[np.asarray(hist, dtype=np.int64)] = False
        for term in intent['exclude']:
            term = term.strip().lower()
            if len(term) >= 4:
                pat = re.compile(r'\b' + re.escape(term) + r's?\b')
                allowed &= ~np.fromiter((bool(pat.search(t)) for t in r.catalog.lower_titles), bool, len(allowed))
        n_allowed = int(allowed.sum())
        if not n_allowed:
            return None
        # Reciprocal-rank fusion of the per-text rankings (selected on the fit half over averaging).
        lists = []
        for rel in rels:
            sc = np.where(allowed, rel, -np.inf)
            m = min(200, n_allowed)
            top = np.argpartition(-sc, m - 1)[:m]
            lists.append(top[np.argsort(-sc[top])])
        fused = rrf(lists)
        pool = np.array(sorted(fused, key=fused.get, reverse=True)[:POOL], dtype=np.int64)
        s = np.full(len(allowed), -np.inf, np.float32)
        s[pool] = [fused[int(i)] for i in pool]
        ce = self.rk.score(intent['query_en'], [doc_text(r.facts, r.catalog.titles[i], i) for i in pool])
        final = z(s[pool]) + PRIOR_WEIGHT * z(r.prior[pool]) + CE_WEIGHT * z(ce)
        personal = None
        if hist and r.ufm is not None:
            personal = r.ufm.score(hist, pool)
            final = final + WEIGHTS['personal'] * z(personal)
        order = np.argsort(-final, kind='stable')
        # Dedupe near-identical listings, then MMR over CLIP text embeddings for variety.
        keep = order[r.catalog.pick(pool[order], min(len(order), self.k * 3), hist)]
        emb = r.clip_text[pool[keep]]
        chosen = [keep[i] for i in mmr(pool[keep], final[keep], emb, self.k, lam=0.8)]
        return dict(pool=pool, final=final, ce=ce, personal=personal, chosen=chosen, n_allowed=n_allowed,
                    rel_pool=s[pool])

    def facets(self, res):
        r, pool, final = self.r, res['pool'], res['final']
        top = pool[np.argsort(-final)[:50]]
        prices = r.facts['price'][top]
        prices = prices[np.isfinite(prices)]
        cats = {}
        for i in top:
            c = r.facts['category'][i].split(' > ')[-1]
            if c:
                cats[c] = cats.get(c, 0) + 1
        return dict(matching=res['n_allowed'],
                    price_range=[round(float(np.percentile(prices, 10)), 2), round(float(np.percentile(prices, 90)), 2)] if len(prices) >= 3 else None,
                    categories=[c for c, _ in sorted(cats.items(), key=lambda x: -x[1])[:4]])

    def suggestions(self, intent, filters, facets, products):
        out = []
        prices = [p['price'] for p in products if p.get('price') is not None]
        if prices:
            cap = round(min(filters.get('max_price') or 1e9, float(np.median(prices))) * 0.7)
            if cap >= 5:
                out.append(f'Rẻ hơn, dưới {cap}$')
        if filters.get('age_years') is None:
            out.append('Cho bé 3 tuổi')
        if 'battery' not in intent['exclude']:
            out.append('Loại không cần pin')
        if not filters.get('min_rating'):
            out.append('Chỉ loại được đánh giá cao')
        for c in facets.get('categories', [])[:2]:
            out.append(f'Xem thêm loại {c}')
        return out[:5]

    # ---------- 3. answer ----------
    def context(self, products, query):
        lines = []
        for i, p in enumerate(products, 1):
            facts = []
            if p.get('price') is not None:
                facts.append(f"price ${p['price']:.2f}")
            if p.get('rating') is not None:
                facts.append(f"rating {p['rating']}/5 ({p['rating_number']} reviews)")
            if p.get('age_min') is not None:
                facts.append(f"age {p['age_min']:g}+")
            if p.get('category'):
                facts.append(p['category'])
            lines.append(f"[{i}] {short_title(p['title'], 90)}\n    {'; '.join(facts)}\n    {compress(p.get('bullets') or p.get('description'), query)}")
        return '\n'.join(lines)

    def ground(self, obj, products, ctx_by_n):
        """Drop reasons that state numbers absent from that product's context (grounding check)."""
        if not isinstance(obj, dict) or not isinstance(obj.get('picks'), list):
            return obj, 0
        dropped = 0
        for pick in obj['picks']:
            try:
                n = int(pick.get('n'))
            except (TypeError, ValueError, AttributeError):
                continue
            why = str(pick.get('why') or '')
            if 1 <= n <= len(products) and _num_tokens(why) - _num_tokens(ctx_by_n[n]):
                p = products[n - 1]
                facts = []
                if p.get('rating') is not None:
                    facts.append(f"đánh giá {p['rating']}/5 ({p['rating_number']} lượt)")
                if p.get('age_min') is not None:
                    facts.append(f"từ {p['age_min']:g} tuổi")
                pick['why'] = ', '.join(facts) or 'khớp với mô tả bạn đưa ra'
                dropped += 1
        return obj, dropped

    def answer(self, message, products, filters, query, note=''):
        ctx = self.context(products, query)
        ctx_by_n = {i: blk for i, blk in enumerate(re.split(r'\n(?=\[\d+\] )', ctx), 1)}
        notes = []
        if filters.get('max_price') is not None:
            notes.append(f"budget up to ${filters['max_price']:g}")
        if filters.get('age_years') is not None:
            notes.append(f"child age {filters['age_years']:g}")
        extra = (f"\nAll products above already satisfy: {', '.join(notes)}." if notes else '') + (f'\nNOTE: {note}' if note else '')
        obj, text, st = self.llm.generate_json([{'role': 'system', 'content': ANSWER_SYSTEM},
                                                {'role': 'user', 'content': f'CONTEXT:\n{ctx}{extra}\n\nUSER REQUEST: {message}'}], 260)
        obj, dropped = self.ground(obj, products, ctx_by_n)
        ans, cited, invalid = render_answer(obj, products)
        if ans is None:
            ans, cited, invalid = citations(text, len(products))
            st['json_failed'] = True
        # Small Qwen models occasionally switch to Chinese mid-sentence; drop those runs.
        cjk = re.compile(r'[\u3040-\u30ff\u4e00-\u9fff\uff00-\uffef]+')
        st['cjk_removed'] = len(cjk.findall(ans))
        ans = re.sub(r'  +', ' ', cjk.sub(' ', ans))
        st['grounding_fixes'] = dropped
        return ans, cited, invalid, st

    # ---------- orchestration ----------
    def run(self, message, state=None, history=()):
        """Generator of events: status / products / answer / done."""
        state, t0 = state or {}, time.monotonic()
        message = message.strip()[:500]
        yield dict(type='status', text='Đang hiểu yêu cầu…')
        intent, s1 = self.understand(message, state)
        if intent['chit_chat'] or not intent['query_en']:
            text, s3 = self.llm.generate([
                {'role': 'system', 'content': "You are a toys & games shopping assistant. Reply briefly in the user's language and invite them to say who the gift is for, age, interests and budget."},
                {'role': 'user', 'content': message}], 120)
            yield dict(type='answer', answer=text, cited=[], intent=intent, state=state,
                       suggestions=['Quà cho bé gái 6 tuổi thích vẽ', 'Board game cho gia đình', 'Đồ chơi gỗ cho bé 2 tuổi'])
            yield dict(type='done', timings=dict(understand=s1['seconds'], answer=s3['seconds'], total=round(time.monotonic() - t0, 2)))
            return
        yield dict(type='status', text='Đang tìm trong 767 nghìn sản phẩm…', intent=intent)
        t1 = time.monotonic()
        filters, relaxed = dict(intent['filters']), []
        res = self.retrieve(intent, filters, history)
        # Corrective RAG: if evidence is weak under the filters, relax the softest constraint and retry.
        for key, change in [('min_rating', None), ('max_price', 1.5), ('age_years', None)]:
            weak = res is None or float(np.max(res['ce'])) < WEAK_CE
            if not weak or filters.get(key) is None:
                continue
            old = filters[key]
            filters[key] = None if change is None else round(old * change, 2)
            relaxed.append(dict(field=key, old=old, new=filters[key]))
            res = self.retrieve(intent, {k: v for k, v in filters.items() if v is not None}, history)
        t_ret = round(time.monotonic() - t1, 2)
        if res is None or not res['chosen']:
            yield dict(type='answer', answer='Mình chưa tìm thấy sản phẩm phù hợp. Bạn mô tả rõ hơn loại đồ chơi, độ tuổi hoặc ngân sách nhé.',
                       cited=[], intent=intent, state=dict(intent=intent), suggestions=[])
            yield dict(type='done', timings=dict(understand=s1['seconds'], retrieve=t_ret, total=round(time.monotonic() - t0, 2)))
            return
        products = []
        for i in res['chosen']:
            row = int(res['pool'][i])
            item = self.r.catalog.item(row)
            item.update(self.r.fact(row), score=round(float(res['final'][i]), 3), rerank=round(float(res['ce'][i]), 3),
                        personal=None if res['personal'] is None else round(float(res['personal'][i]), 3))
            products.append(item)
        facets = self.facets(res)
        weak = float(np.max(res['ce'])) < WEAK_CE
        yield dict(type='products', products=products, facets=facets, relaxed=relaxed, weak=weak,
                   personalised=res['personal'] is not None, retrieve_seconds=t_ret)
        yield dict(type='status', text='Đang soạn câu trả lời…')
        note = ''
        if relaxed:
            note = 'Some filters were relaxed because nothing matched well: ' + ', '.join(f"{x['field']} {x['old']}→{x['new']}" for x in relaxed) + '. Mention this briefly in intro.'
        if weak:
            note += ' The matches are weak; say so in intro and ask one clarifying question in tip.'
        ans, cited, invalid, s3 = self.answer(message, products, filters, intent['query_en'], note.strip())
        yield dict(type='answer', answer=ans, cited=cited, invalid_citations=invalid, intent=dict(intent, filters={k: v for k, v in filters.items() if v is not None}),
                   state=dict(intent=dict(intent, filters={k: v for k, v in filters.items() if v is not None})),
                   suggestions=self.suggestions(intent, filters, facets, products))
        yield dict(type='done', timings=dict(understand=s1['seconds'], retrieve=t_ret, answer=s3['seconds'],
                                             grounding_fixes=s3.get('grounding_fixes', 0), total=round(time.monotonic() - t0, 2)))

    def chat(self, message, state=None, history=()):
        out = {}
        for ev in self.run(message, state, history):
            out.update({k: v for k, v in ev.items() if k != 'type'})
        return out
