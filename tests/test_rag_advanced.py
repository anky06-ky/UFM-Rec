"""Model-free tests for advanced-RAG helpers and the v2 grounding step."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rag'))
from advanced import rrf, mmr, compress  # noqa: E402
from pipeline_v2 import AdvancedRAG, _num_tokens  # noqa: E402


def test_rrf_rewards_agreement():
    s = rrf([np.array([1, 2, 3]), np.array([2, 1, 4])])
    assert max(s, key=s.get) in (1, 2) and s[1] == s[2] and s[3] > 0 and s[4] > 0 and s[1] > s[3]


def test_mmr_skips_near_duplicate():
    emb = np.array([[1, 0], [0.999, 0.045], [0, 1], [0.7, 0.7]], np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    chosen = mmr(np.arange(4), np.array([1.0, 0.99, 0.8, 0.0]), emb, 2, lam=0.6)
    assert chosen == [0, 2]


def test_compress_keeps_query_relevant_fragment():
    text = 'Made of premium plastic | Remote control with 2.4G for kids | Comes in a gift box'
    out = compress(text, 'remote control dinosaur', 60)
    assert out.startswith('Remote control') and len(out) <= 80


def test_ground_replaces_reason_with_invented_number():
    rag = AdvancedRAG.__new__(AdvancedRAG)
    products = [dict(rating=4.5, rating_number=120, age_min=3.0), dict(rating=None, rating_number=0, age_min=None)]
    ctx = {1: '[1] Toy\n    price $12.99; rating 4.5/5 (120 reviews); age 3+', 2: '[2] Other\n    soft plush'}
    obj = dict(picks=[dict(n=1, why='giá 12.99$, hợp bé 3 tuổi'), dict(n=2, why='phù hợp cho bé 5 tuổi')])
    obj, fixes = rag.ground(obj, products, ctx)
    assert fixes == 1 and obj['picks'][0]['why'] == 'giá 12.99$, hợp bé 3 tuổi' and '5' not in obj['picks'][1]['why']


def test_num_tokens():
    assert _num_tokens('4.5/5 (120 reviews), $12,99') == {'4.5', '5', '120', '12,99'}
