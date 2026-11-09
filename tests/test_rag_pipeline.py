"""Model-free tests for the RAG glue (constraint parsing, grounded answer rendering)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rag'))
import pytest
from pipeline import parse_constraints, render_answer, citations, short_title

P = [dict(title='Dinosaur Toy A', price=12.5), dict(title='Robot B', price=None), dict(title='Puzzle C', price=30.0)]


@pytest.mark.parametrize('text,expected', [
    ('quà cho bé 5 tuổi dưới 30$', dict(age_years=5.0, max_price=30.0)),
    ('board game dưới 500k', dict(max_price=20.0)),
    ('đồ chơi dưới 1 triệu', dict(max_price=40.0)),
    ('toy for 8 years old under $25', dict(age_years=8.0, max_price=25.0)),
    ('đồ chơi cho bé 18 tháng', dict(age_years=1.5)),
    ('lego được đánh giá cao', dict(min_rating=4.3)),
    ('xe điều khiển từ xa', dict()),
])
def test_parse_constraints(text, expected):
    assert parse_constraints(text) == expected


def test_age_not_taken_as_min_price():
    assert 'min_price' not in parse_constraints('đồ chơi cho trẻ trên 3 tuổi')


def test_refine_cheaper_keeps_previous_and_lowers_budget():
    c = parse_constraints('có cái nào rẻ hơn không', dict(max_price=30.0, age_years=5.0))
    assert c == dict(max_price=21.0, age_years=5.0)


def test_render_answer_uses_catalog_facts_and_drops_invalid():
    obj = dict(intro='Gợi ý:', picks=[dict(n=1, why='hợp tuổi'), dict(n=9, why='x'), dict(n=1, why='dup'), dict(n=2)], tip='')
    text, cited, invalid = render_answer(obj, P)
    assert cited == [1, 2] and invalid == [9]
    assert '- [1] Dinosaur Toy A ($12.50): hợp tuổi' in text and '- [2] Robot B' in text and '$' not in text.split('\n')[2]


def test_render_answer_rejects_non_json_shape():
    assert render_answer(None, P)[0] is None and render_answer(dict(picks='x'), P)[0] is None


def test_citations_strip_unknown_numbers():
    clean, cited, invalid = citations('Xem [2] và [7], cả [2].', 3)
    assert cited == [2] and invalid == [7] and '[7]' not in clean


def test_short_title():
    assert len(short_title('A' * 30 + ' - ' + 'B' * 80)) <= 70
