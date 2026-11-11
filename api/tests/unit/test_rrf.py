from app.models import Chunk
from app.retrieval.search import reciprocal_rank_fusion


def chunks(*ids: int) -> list[Chunk]:
    return [Chunk(id=chunk_id, document_id=1) for chunk_id in ids]


def ids(ranking: list[Chunk]) -> list[int]:
    return [chunk.id for chunk in ranking]


def test_chunk_found_by_both_searches_beats_a_chunk_found_by_one():
    """Chunk 2 is ranked 2nd in both individual searches. Yet RRF ranks it 1st."""
    fused = reciprocal_rank_fusion([chunks(1, 2), chunks(3, 2)])
    assert ids(fused)[0] == 2


def test_earlier_rank_scores_higher():
    assert ids(reciprocal_rank_fusion([chunks(1, 2, 3)])) == [1, 2, 3]


def test_each_chunk_appears_once():
    fused = reciprocal_rank_fusion([chunks(1, 2), chunks(2, 1)])
    assert sorted(ids(fused)) == [1, 2]


def test_empty_rankings_give_nothing():
    assert reciprocal_rank_fusion([[], []]) == []
