import pytest

from app.evaluation import calculate_reciprocal_rank, calculate_hit_at_k, calculate_recall_at_k


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

def test_calculate_recall_at_k_returns_partial_recall():
    results = [
        {"chunk_index": 10},
        {"chunk_index": 50},
        {"chunk_index": 30},
    ]

    relevant_chunk_indices = [10, 20, 30, 40]

    # TODO: calculate_recall_at_k 호출
    crk = calculate_recall_at_k(results=results,
                                relevant_chunk_indices=relevant_chunk_indices,
                                k=3)
    
    # TODO: assert로 결과 검증
    assert crk == 0.5

def test_calculate_recall_at_k_ignores_duplicate_chunks():
    results = [
        {"chunk_index": 10},
        {"chunk_index": 10},
        {"chunk_index": 20},
        {"chunk_index": 30},
    ]

    relevant_chunk_indices = [10, 20, 30]

    # TODO: 직접 작성
    crk = calculate_recall_at_k(results=results,
                                relevant_chunk_indices=relevant_chunk_indices,
                                k=3)
    
    # TODO: assert로 결과 검증
    assert crk == pytest.approx(2/3)

def test_calculate_recall_at_k_raises_value_error_when_k_is_not_positive():
    with pytest.raises(ValueError):
        calculate_recall_at_k(
            results=[],
            relevant_chunk_indices=[10],
            k=0,
        )
