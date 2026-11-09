"""Advanced-RAG building blocks: cross-encoder re-ranking, reciprocal-rank fusion, MMR diversity,
contextual compression of product text."""
import re
from pathlib import Path
import numpy as np

from llm import HF_HOME
HF = HF_HOME / 'hub'
RERANKERS = {
    'minilm': ('cross-encoder/ms-marco-MiniLM-L-6-v2', '233902d25c440f23af6f7d6e94d2946bac0bee0a'),
    'bge': ('BAAI/bge-reranker-base', '2cfc18c9415c912f9d8155881c133215df768a70'),
}


class Reranker:
    def __init__(self, name='minilm', device='auto', max_length=256):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        repo, rev = RERANKERS[name]
        kw = dict(revision=rev, cache_dir=HF, trust_remote_code=False)
        self.tok = AutoTokenizer.from_pretrained(repo, **kw)
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 1.5e9 else 'cpu'
        self.model = AutoModelForSequenceClassification.from_pretrained(repo, **kw).to(device).eval()
        if device == 'cuda':
            self.model.half()
        self.torch, self.device, self.max_length, self.name = torch, device, max_length, name

    def score(self, query, docs, batch=64):
        out = []
        with self.torch.inference_mode():
            for s in range(0, len(docs), batch):
                enc = self.tok([query] * len(docs[s:s + batch]), docs[s:s + batch], padding=True, truncation='only_second',
                               max_length=self.max_length, return_tensors='pt').to(self.device)
                out.append(self.model(**enc).logits[:, 0].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)


def doc_text(facts, title, r, n=300):
    parts = [title, facts['category'][r], (facts['bullets'][r] or facts['description'][r])[:n]]
    return ' | '.join(p for p in parts if p)


def rrf(rank_lists, k=60):
    """rank_lists: list of arrays of item ids in rank order. Returns dict id -> fused score."""
    score = {}
    for lst in rank_lists:
        for pos, i in enumerate(lst):
            score[int(i)] = score.get(int(i), 0.0) + 1.0 / (k + pos + 1)
    return score


def mmr(candidates, relevance, emb, k, lam=0.75):
    """Maximal marginal relevance over unit-norm embeddings; returns indices into candidates."""
    rel = (relevance - relevance.min()) / (np.ptp(relevance) or 1)
    chosen, rest = [], list(range(len(candidates)))
    sims = emb @ emb.T
    while rest and len(chosen) < k:
        if not chosen:
            j = max(rest, key=lambda i: rel[i])
        else:
            j = max(rest, key=lambda i: lam * rel[i] - (1 - lam) * max(sims[i, c] for c in chosen))
        chosen.append(j)
        rest.remove(j)
    return chosen


_WORD = re.compile(r'[a-z0-9]+')


def compress(text, query, max_chars=220):
    """Keep the bullet/sentence fragments that share most words with the query (contextual compression)."""
    if not text:
        return ''
    q = set(_WORD.findall(query.lower())) - {'for', 'the', 'and', 'with', 'toy', 'toys', 'kids', 'a', 'of'}
    frags = [f.strip() for f in re.split(r'\s\|\s|(?<=[.!?])\s+', text) if len(f.strip()) > 15]
    if not frags:
        return text[:max_chars]
    scored = sorted(enumerate(frags), key=lambda x: (-len(q & set(_WORD.findall(x[1].lower()))), x[0]))
    out, used = [], 0
    for _, f in scored:
        f = f if len(f) <= 160 else f[:159].rsplit(' ', 1)[0] + '…'
        if used + len(f) > max_chars and out:
            break
        out.append(f)
        used += len(f) + 2
    return '; '.join(out)
