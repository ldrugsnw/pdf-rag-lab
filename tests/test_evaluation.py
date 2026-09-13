import pytest

from evaluation import calculate_reciprocal_rank, calculate_hit_at_k


def test_calculate_reciprocal_rank_uses_first_relevant_result():
    results = [
        {"chunk_index": 10},
        {"chunk_index": 13},
        {"chunk_index": 20},
    ]

    rr = calculate_reciprocal_rank(
        results=results,
        relevant_chunk_indices=[13, 20],
    )

    assert rr == pytest.approx(0.5)


def test_calculate_reciprocal_rank_returns_zero_when_no_result_is_relevant():
    results = [
        {"chunk_index": 10},
        {"chunk_index": 11},
    ]

    rr = calculate_reciprocal_rank(
        results=results,
        relevant_chunk_indices=[13],
    )

    assert rr == 0.0


def test_calculate_hit_at_k_returns_one_when_there_is_answer():
    results = [
        {"chunk_index": 10},
        {"chunk_index": 11},
        {"chunk_index": 13},
        {"chunk_index": 20},
    ]

    hak = calculate_hit_at_k(
       results=results,
        relevant_chunk_indices=[13],
        k=3,
    )    

    assert hak == 1.0

def test_calculate_hit_at_k_returns_zero_when_there_is_no_answer():
    results = [
        {"chunk_index": 10},
        {"chunk_index": 11},
        {"chunk_index": 13},
        {"chunk_index": 20},
    ]

    hak = calculate_hit_at_k(
       results=results,
        relevant_chunk_indices=[20],
        k=3,
    )    

    assert hak == 0.0


def test_calculate_hit_at_k_raises_value_error_when_k_is_not_positive():
    with pytest.raises(ValueError):
        calculate_hit_at_k(
            results=[],
            relevant_chunk_indices=[13],
            k=0,
        )