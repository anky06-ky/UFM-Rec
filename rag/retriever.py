"""Query-driven retrieval and re-ranking for the RAG layer.

Stage 1 (recall): a free-text English query is matched against the whole catalog (767k items) with
  the project's TF-IDF vectorizer and with CLIP ViT-B/32 text features against the stored CLIP
  text and image embeddings of every product. Hard filters (price, age, rating) apply here.
Stage 2 (re-rank): query relevance + recent popularity (train interactions only) + UFM v2
  personalisation when the user supplies a purchase history.
The LLM never chooses products: it only rewrites the request and explains this ranking.
"""
import csv, gzip, json, re, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import joblib
import pyarrow.parquet as pq
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from recommend_catalog import Catalog, DATA  # noqa: E402
from common import sha256  # noqa: E402

CLIP_REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
# Relevance blend (CLIP text/image vs TF-IDF) and re-rank weights. Chosen on the synthetic query
# set (fit half) by rag/tune_weights.py; audit-half numbers in reports/rag_20261109/weights_wide.json.
WEIGHTS = dict(lex=0.1, clip_text=1.0, clip_image=1.0, prior=0.5, personal=0.5)
POOL = 400


def z(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def recent_counts(data, days):
    """Interactions per item in the last `days` days of the train split (cached)."""
    cache = data / f'rag_facts_v1/recent_{days}d.npy'
    if cache.is_file():
        return np.load(cache)
    manifest = json.loads((data / 'split_manifest.json').read_text())
    end = int(datetime.fromisoformat(manifest['validation_start_utc']).timestamp() * 1000)
    start = end - days * 86400000
    lookup = {}
    with gzip.open(data / 'content/items.csv.gz', 'rt', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            lookup[r['parent_asin']] = int(r['row_index'])
    out = np.zeros(len(lookup), dtype=np.int32)
    with gzip.open(data / 'interactions_train.csv.gz', 'rt', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if int(r['timestamp']) >= start:
                out[lookup[r['parent_asin']]] += 1
    np.save(cache, out)
    return out


class UFMv2Scorer:
    """Scores candidate rows for one history with the UFM v2 checkpoint (CPU)."""

    def __init__(self, catalog, run, chunk=8192):
        import torch
        self.torch, self.chunk = torch, chunk
        run = Path(run)
        config = json.loads((run / 'config.json').read_text())
        src = ROOT / 'src/ufm_model_v2.py'
        if sha256(src) != config['code_sha256']['ufm_model_v2.py']:
            raise ValueError('ufm_model_v2.py changed since training.')
        from ufm_model_v2 import UFMConfig, UFMRec
        state = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
        self.cfg = UFMConfig(**state['model_config'])
        if self.cfg.items != len(catalog.asins):
            raise ValueError('Model/catalog mismatch.')
        self.model = UFMRec(self.cfg).eval()
        self.model.load_state_dict(state['model'])
        self.epoch = state.get('epoch')
        del state
        feats = Path(config['features'])
        tables = [torch.from_numpy(np.load(feats / n, mmap_mode='r').copy()) for n in ['text.npy', 'image.npy', 'modalities.npy']]
        extra = torch.from_numpy(np.load(Path(config['extra']) / 'lsa.npy').astype(np.float32))
        counts = torch.from_numpy(np.concatenate([[0], catalog.counts]).astype(np.int64))
        torch.set_num_threads(8)
        with torch.inference_mode():
            ids = torch.arange(self.cfg.items + 1)
            parts = [self.model.semantic_items(ids[i:i + chunk], *tables, extra) for i in range(0, len(ids), chunk)]
        semantic, present = (torch.cat(x) for x in zip(*parts))
        self.model.semantic_items = lambda ids, *a, **k: (semantic[ids], present[ids])
        self.tables = (tables[0][:1], tables[1][:1], tables[2], counts)  # text/image unused after caching
        del tables, extra

    def score(self, history_rows, rows):
        torch = self.torch
        h = np.zeros((1, self.cfg.history_size), dtype=np.int64)
        past = history_rows[-self.cfg.history_size:]
        h[0, :len(past)] = np.asarray(past, dtype=np.int64) + 1
        h = torch.from_numpy(h)
        out = np.empty(len(rows), dtype=np.float64)
        with torch.inference_mode():
            for s in range(0, len(rows), self.chunk):
                part = np.asarray(rows[s:s + self.chunk], dtype=np.int64)
                o = self.model(h, torch.from_numpy(part[None] + 1), *self.tables)
                out[s:s + len(part)] = o['scores'][0].numpy()
        return out


class QueryRetriever:
    def __init__(self, data=DATA, ufm_run=ROOT / 'runs/ufm_v2', catalog=None, log=print):
        t0 = time.monotonic()
        self.data = Path(data)
        self.catalog = catalog or Catalog(self.data)
        n = len(self.catalog.asins)
        self.matrix = sparse.load_npz(self.data / 'content/tfidf_all.npz').tocsr()
        self.vectorizer = joblib.load(self.data / 'content/vectorizer.joblib')
        feats = self.data / 'foundation_clip_b32_v1'
        # Row 0 is padding; keep float16 to halve memory, matmul in float32 chunks.
        self.clip_text = np.load(feats / 'text.npy', mmap_mode='r')[1:]
        self.clip_image = np.load(feats / 'image.npy', mmap_mode='r')[1:]
        self.clip_text, self.clip_image = self.clip_text.astype(np.float32), self.clip_image.astype(np.float32)
        mods = np.load(feats / 'modalities.npy')[1:]
        self.has_text, self.has_image = mods[:, 0] > 0, mods[:, 1] > 0
        facts = pq.read_table(self.data / 'rag_facts_v1/facts.parquet')
        self.facts = {c: facts[c].to_numpy(zero_copy_only=False) for c in facts.column_names}
        if len(self.facts['row_index']) != n:
            raise ValueError('Facts table misaligned with catalog.')
        recent = recent_counts(self.data, 180)
        self.prior = z(np.log1p(recent) + 0.5 * np.log1p(self.catalog.counts))
        log(f'retriever tables ready in {time.monotonic() - t0:.0f}s')
        self.clip = None
        self.ufm = None
        if ufm_run:
            self.ufm = UFMv2Scorer(self.catalog, ufm_run)
            log(f'UFM v2 (epoch {self.ufm.epoch}) ready in {time.monotonic() - t0:.0f}s')

    def encode(self, text):
        if self.clip is None:
            from ufm_model_v2 import FrozenCLIPEncoder
            self.clip = FrozenCLIPEncoder(CLIP_REVISION, 'cpu', cache_dir=ROOT / 'tmp/hf_ufm_models')
        return self.clip.encode_text([text])[0].numpy().astype(np.float32)

    def _dense(self, table, q, chunk=131072):
        out = np.empty(len(table), dtype=np.float32)
        for s in range(0, len(table), chunk):
            out[s:s + chunk] = table[s:s + chunk] @ q
        return out

    def mask(self, f):
        m = np.ones(len(self.catalog.asins), dtype=bool)
        price, age, rating = self.facts['price'], self.facts['age_min'], self.facts['rating']
        if f.get('max_price') is not None:
            m &= np.nan_to_num(price, nan=np.inf) <= float(f['max_price'])
        if f.get('min_price') is not None:
            m &= np.nan_to_num(price, nan=-np.inf) >= float(f['min_price'])
        if f.get('age_years') is not None:
            # Unknown age is allowed; known minimum age must not exceed the child's age.
            m &= np.nan_to_num(age, nan=0) <= float(f['age_years']) + 0.01
        if f.get('min_rating') is not None:
            m &= np.nan_to_num(rating, nan=0) >= float(f['min_rating'])
        return m

    def relevance(self, query, weights=WEIGHTS):
        q = self.vectorizer.transform([query])
        lex = np.asarray((self.matrix @ q.T).todense()).ravel()
        qe = self.encode(query)
        ct = np.where(self.has_text, self._dense(self.clip_text, qe), 0)
        ci = np.where(self.has_image, self._dense(self.clip_image, qe), 0)
        return weights.get('lex', 1.0) * z(lex) + weights['clip_text'] * z(ct) + weights['clip_image'] * z(ci), dict(lex=lex, clip_text=ct, clip_image=ci)

    def relevance_many(self, texts, weights=WEIGHTS, chunk=131072):
        """Blended relevance for several query texts in one pass over the embedding tables -> [T, N]."""
        if self.clip is None:
            self.encode('warm up')
        q = self.vectorizer.transform(texts)
        lex = (self.matrix @ q.T).toarray().T.astype(np.float32)
        qe = self.clip.encode_text(list(texts)).numpy().astype(np.float32).T  # [512, T]
        ct = np.empty((len(texts), len(self.clip_text)), np.float32)
        ci = np.empty_like(ct)
        for s in range(0, len(self.clip_text), chunk):
            ct[:, s:s + chunk] = (self.clip_text[s:s + chunk] @ qe).T
            ci[:, s:s + chunk] = (self.clip_image[s:s + chunk] @ qe).T
        ct[:, ~self.has_text] = 0
        ci[:, ~self.has_image] = 0
        zr = lambda a: (a - a.mean(1, keepdims=True)) / np.where(a.std(1, keepdims=True) > 0, a.std(1, keepdims=True), 1)
        return weights.get('lex', 1.0) * zr(lex) + weights['clip_text'] * zr(ct) + weights['clip_image'] * zr(ci)

    def search(self, query, filters=None, history=(), exclude_terms=(), k=8, weights=WEIGHTS):
        """Return k products for a query. history: list of ASINs (personalisation, excluded from results)."""
        t0 = time.monotonic()
        filters = filters or {}
        rel, parts = self.relevance(query, weights)
        allowed = self.mask(filters)
        hist = self.catalog.rows(list(history)[-20:]) if history else []
        allowed[np.asarray(hist, dtype=np.int64)] = False
        for term in exclude_terms:
            term = term.strip().lower()
            if len(term) >= 4:
                # Drop items whose title contains the excluded whole word (e.g. "battery").
                pat = re.compile(r'\b' + re.escape(term) + r's?\b')
                hit = np.fromiter((bool(pat.search(t)) for t in self.catalog.lower_titles), bool, len(allowed))
                allowed &= ~hit
        rows = np.flatnonzero(allowed)
        if not len(rows):
            return [], dict(candidates=0, seconds=round(time.monotonic() - t0, 2))
        pool = rows[np.argsort(-rel[rows])[:POOL]]
        final = z(rel[pool]) + weights['prior'] * z(self.prior[pool])
        personal = None
        if hist and self.ufm is not None and weights['personal'] > 0:
            personal = self.ufm.score(hist, pool)
            final = final + weights['personal'] * z(personal)
        order = np.argsort(-final, kind='stable')
        picked = order[self.catalog.pick(pool[order], k, hist)]
        out = []
        for i in picked:
            r = int(pool[i])
            item = self.catalog.item(r)
            item.update(self.fact(r), score=round(float(final[i]), 4), relevance=round(float(rel[r]), 4),
                        lexical=round(float(parts['lex'][r]), 4), clip_text=round(float(parts['clip_text'][r]), 4),
                        personal=None if personal is None else round(float(personal[i]), 4))
            out.append(item)
        return out, dict(candidates=int(len(rows)), pool=int(len(pool)), personalised=personal is not None,
                         seconds=round(time.monotonic() - t0, 2))

    def fact(self, r):
        f = self.facts
        num = lambda v: None if not np.isfinite(v) else round(float(v), 2)
        return dict(price=num(f['price'][r]), rating=num(f['rating'][r]), rating_number=int(f['rating_number'][r]),
                    age_min=num(f['age_min'][r]), category=f['category'][r], store=f['store'][r],
                    bullets=f['bullets'][r], description=f['description'][r])
