"""RAG pipeline: LLM request understanding -> catalog retrieval + UFM re-rank -> grounded LLM answer."""
import json, re, time

VND_PER_USD = 25000

INTENT_SYSTEM = """You turn a shopper's message about toys and games (Vietnamese or English) into a JSON search request.
If PREVIOUS is given and the new message refines it ("cheaper", "rẻ hơn", "for girls", "không dùng pin"), merge them; if it asks for something new, start over.
Return ONLY one JSON object with these keys:
  "query_en": short English product search query (3-10 words: product type + key attributes such as theme, material, piece count, players; no price, no age),
  "exclude": list of English words to avoid in product titles (e.g. ["battery"]; translate Vietnamese "pin" to "battery"), usually [],
  "refine": true if the message only adjusts PREVIOUS (cheaper, other colour, remote control...), false for a new request,
  "chit_chat": true only if the message is not about finding a product.
Examples:
User: quà cho bé trai 5 tuổi thích khủng long, dưới 30$
{"query_en": "dinosaur toy for boys", "exclude": [], "refine": false, "chit_chat": false}
User: bộ xếp hình gỗ cho bé 2 tuổi, không cần pin
{"query_en": "wooden building blocks for toddlers", "exclude": ["battery"], "refine": false, "chit_chat": false}
PREVIOUS: {"query_en": "dinosaur toy for boys"}  User: có loại nào điều khiển từ xa không?
{"query_en": "remote control dinosaur toy", "exclude": [], "refine": true, "chit_chat": false}
PREVIOUS: {"query_en": "remote control dinosaur toy"}  User: còn đồ chơi nấu ăn cho bé gái thì sao?
{"query_en": "kids pretend play kitchen cooking set for girls", "exclude": [], "refine": false, "chit_chat": false}
User: puzzle 1000 mảnh cho người lớn
{"query_en": "1000 piece jigsaw puzzle for adults", "exclude": [], "refine": false, "chit_chat": false}
User: xin chào
{"query_en": "", "exclude": [], "refine": false, "chit_chat": true}"""

INTENT_SYSTEM_V2 = """You turn a shopper's message about toys and games (Vietnamese or English) into a JSON search plan.
If PREVIOUS is given and the new message only adjusts it ("cheaper", "rẻ hơn", "loại điều khiển từ xa", "không dùng pin"), merge them; otherwise start over.
Return ONLY one JSON object with keys:
  "query_en": main English search query, 3-10 words (product type + key attributes such as theme, material, piece count, players; no price, no age),
  "alt_queries": 2 different English phrasings or related product types a store might use for the same need,
  "hyde": one English sentence describing an ideal matching product as a catalog listing would (type, features, who it is for),
  "exclude": English words to avoid in product titles (translate "pin" to "battery"), usually [],
  "refine": true if the message only adjusts PREVIOUS, else false,
  "chit_chat": true only if the message is not about finding a product.
Example:
User: quà cho bé trai 5 tuổi thích khủng long, dưới 30$
{"query_en": "dinosaur toy for boys", "alt_queries": ["dinosaur figures playset", "take apart dinosaur building toy"], "hyde": "A set of realistic dinosaur figures with a play mat, safe plastic, great gift for boys ages 3-6.", "exclude": [], "refine": false, "chit_chat": false}
User: xin chào
{"query_en": "", "alt_queries": [], "hyde": "", "exclude": [], "refine": false, "chit_chat": true}"""

ANSWER_SYSTEM = """You are a shopping assistant for a toys & games catalog. The user's request and numbered products (CONTEXT) are given.
Choose the 3-4 products that best fit the request, best first.
Return ONLY one JSON object:
{"intro": "one short friendly sentence", "picks": [{"n": 2, "why": "why it fits, max 20 words"}], "tip": "one short tip to refine, or empty"}
Rules: write intro, why and tip in the user's language (Vietnamese if the request is Vietnamese).
"why" may use only facts from CONTEXT (features, age, rating, price) and must not contradict them; never invent facts.
Only pick numbers that exist in CONTEXT. If nothing fits, return an empty picks list and explain in intro."""


def parse_constraints(text, previous=None):
    """Price / age / rating from the message with regexes (deterministic, LLMs mis-convert numbers)."""
    c = dict(previous or {})
    t = text.lower().replace(',', '.')
    num = r'(\d+(?:\.\d+)?)'

    def money(v, unit):
        v = float(v)
        unit = (unit or '').strip()
        if unit in ('k', 'nghìn', 'ngàn', 'ng'):
            return v * 1000 / VND_PER_USD
        if unit in ('tr', 'triệu', 'm'):
            return v * 1e6 / VND_PER_USD
        if unit in ('đ', 'vnd', 'vnđ', 'đồng') or v >= 10000:
            return v / VND_PER_USD
        return v
    unit = r'\s*(k|nghìn|ngàn|tr|triệu|đ|vnđ|vnd|đồng|\$|usd|đô)?'
    m = re.search(r'(?:dưới|tối đa|không quá|<=?|under|below|less than|max|up to)\s*\$?\s*' + num + unit, t)
    if m:
        c['max_price'] = round(money(m.group(1), m.group(2)), 2)
    m = re.search(r'(?:trên|từ|over|above|at least|more than)\s*\$?\s*' + num + unit + r'(?!\s*(?:tuổi|year|yr|tháng|month))', t)
    if m and (m.group(2) or '$' in t[max(0, m.start() - 1):m.end()]):
        c['min_price'] = round(money(m.group(1), m.group(2)), 2)
    m = re.search(num + r'\s*(?:tuổi|years? old|yo\b|year-old)', t)
    if m:
        c['age_years'] = float(m.group(1))
    m = re.search(num + r'\s*(?:tháng|months? old)', t)
    if m:
        c['age_years'] = round(float(m.group(1)) / 12, 2)
    if re.search(r'rẻ hơn|cheaper|less expensive', t) and c.get('max_price') and not re.search(r'dưới|under|below', t):
        c['max_price'] = round(c['max_price'] * 0.7, 2)
    if re.search(r'đánh giá cao|tốt nhất|highly rated|best rated|top rated', t):
        c['min_rating'] = 4.3
    return c


def context_block(products):
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
        bullets = (p.get('bullets') or p.get('description') or '')[:300]
        lines.append(f"[{i}] {p['title'][:150]}\n    {'; '.join(facts)}\n    {bullets}")
    return '\n'.join(lines)


def short_title(t, n=70):
    t = re.split(r'\s[-|:,(]\s?|, ', t)[0] if len(t) > n else t
    return t if len(t) <= n else t[:n - 1].rsplit(' ', 1)[0] + '…'


def render_answer(obj, products):
    """Turn the LLM's JSON choice into text; titles and prices come from the catalog, not the LLM."""
    if not isinstance(obj, dict) or not isinstance(obj.get('picks'), list):
        return None, [], []
    lines, cited, invalid = [str(obj.get('intro') or '').strip()], [], []
    for pick in obj['picks'][:5]:
        try:
            k = int(pick.get('n'))
        except (TypeError, ValueError, AttributeError):
            continue
        if not 1 <= k <= len(products):
            invalid.append(k); continue
        if k in cited:
            continue
        cited.append(k)
        p = products[k - 1]
        price = f" (${p['price']:.2f})" if p.get('price') is not None else ''
        why = str(pick.get('why') or '').strip()
        lines.append(f"- [{k}] {short_title(p['title'])}{price}: {why}" if why else f"- [{k}] {short_title(p['title'])}{price}")
    tip = str(obj.get('tip') or '').strip()
    if tip:
        lines.append(tip)
    return '\n'.join(l for l in lines if l), cited, invalid


def citations(answer, n):
    cited = []
    for m in re.finditer(r'\[(\d+)\]', answer):
        k = int(m.group(1))
        if 1 <= k <= n and k not in cited:
            cited.append(k)
    invalid = sorted({int(m.group(1)) for m in re.finditer(r'\[(\d+)\]', answer)} - set(range(1, n + 1)))
    clean = re.sub(r'\[(\d+)\]', lambda m: m.group(0) if 1 <= int(m.group(1)) <= n else '', answer)
    return clean, cited, invalid


class RAGPipeline:
    def __init__(self, retriever, llm, k=6):
        self.retriever, self.llm, self.k = retriever, llm, k

    def understand(self, message, state):
        prev = state.get('intent') or {}
        user = message if not prev.get('query_en') else f"PREVIOUS: {json.dumps({'query_en': prev['query_en']}, ensure_ascii=False)}  User: {message}"
        obj, raw, stats = self.llm.generate_json(
            [{'role': 'system', 'content': INTENT_SYSTEM}, {'role': 'user', 'content': user}], 80)
        if not isinstance(obj, dict) or not isinstance(obj.get('query_en', ''), str):
            obj = dict(query_en=message, exclude=[], chit_chat=False, fallback=True)
        query = re.sub(r'\b(under|below|less than|over|above)\s*\$?\s*\d+\S*|\$\s*\d+\S*|\d+\s*\$|\d+\s*(?:years?|yrs?)(?:\s*old)?', ' ',
                       str(obj.get('query_en') or ''), flags=re.I)
        refine = bool(obj.get('refine')) and bool(prev.get('query_en'))
        intent = dict(query_en=re.sub(r'\s+', ' ', query).strip(), exclude=[str(x) for x in obj.get('exclude') or []][:5],
                      refine=refine, chit_chat=bool(obj.get('chit_chat')))
        intent['filters'] = parse_constraints(message, prev.get('filters') if refine else None)
        return intent, stats

    def answer(self, message, products, filters):
        notes = []
        if filters.get('max_price') is not None:
            notes.append(f"budget up to ${filters['max_price']:g}")
        if filters.get('age_years') is not None:
            notes.append(f"child age {filters['age_years']:g}")
        note = f"\nAll products above already satisfy: {', '.join(notes)}." if notes else ''
        obj, text, stats = self.llm.generate_json([
            {'role': 'system', 'content': ANSWER_SYSTEM},
            {'role': 'user', 'content': f'CONTEXT:\n{context_block(products)}{note}\n\nUSER REQUEST: {message}'}], 260)
        answer, cited, invalid = render_answer(obj, products)
        if answer is None:  # JSON failed: keep the free text but drop citations to unknown numbers
            answer, cited, invalid = citations(text, len(products))
            stats['json_failed'] = True
        return answer, cited, invalid, stats

    def chat(self, message, state=None, history=()):
        state = state or {}
        t0 = time.monotonic()
        message = message.strip()[:500]
        intent, s1 = self.understand(message, state)
        if intent['chit_chat'] or not intent['query_en']:
            text, s3 = self.llm.generate([
                {'role': 'system', 'content': 'You are a toys & games shopping assistant. Reply briefly in the user\'s language and invite them to describe who the gift is for, age, interests and budget.'},
                {'role': 'user', 'content': message}], 120, temperature=0.0)
            return dict(answer=text, products=[], cited=[], intent=intent, state=state,
                        timings=dict(understand=s1['seconds'], answer=s3['seconds'], total=round(time.monotonic() - t0, 2)))
        products, rstats = self.retriever.search(intent['query_en'], intent['filters'], history,
                                                 intent['exclude'], k=self.k)
        if not products:
            answer = ('Không tìm thấy sản phẩm khớp với các điều kiện (giá, độ tuổi, đánh giá). '
                      'Bạn thử nới điều kiện, ví dụ tăng ngân sách hoặc bỏ bớt yêu cầu.')
            return dict(answer=answer, products=[], cited=[], intent=intent, state=dict(intent=intent),
                        retrieval=rstats, timings=dict(understand=s1['seconds'], total=round(time.monotonic() - t0, 2)))
        answer, cited, invalid, s3 = self.answer(message, products, intent['filters'])
        return dict(answer=answer, products=products, cited=cited, invalid_citations=invalid, intent=intent,
                    state=dict(intent=intent), retrieval=rstats,
                    timings=dict(understand=s1['seconds'], retrieve=rstats['seconds'], answer=s3['seconds'],
                                 total=round(time.monotonic() - t0, 2)))
