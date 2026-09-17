import pytest

from app.reranker import rerank_results


def test_rerank_results_orders_candidates_by_rerank_score():
    candidates = [
        {
            "chunk_index": 10,
            "score": 0.6077,
            "text": "general RAG architecture",
        },
        {
            "chunk_index": 27,
            "score": 0.6028,
            "text": "experiment results",
        },
        {
            "chunk_index": 13,
            "score": 0.5237,
            "text": "the retriever is based on DPR",
        },
    ]

    rerank_scores = [
        {"chunk_index": 27, "score": 0.2},
        {"chunk_index": 10, "score": 0.4},
        {"chunk_index": 13, "score": 0.9},
    ]

    results = rerank_results(
        candidates=candidates,
        rerank_scores=rerank_scores,
        top_k=2,
    )

    assert len(results) == 2
    
    assert results[0]["chunk_index"] == 13
    assert results[0]["rerank_score"] == 0.9

    assert results[1]["chunk_index"] == 10
    assert results[1]["rerank_score"] == 0.4

    assert results[0]["score"] == 0.5237 
    # 기존의 score가 날라가는 않았는가?


def test_rerank_results_raises_error_when_lengths_are_different():
    with pytest.raises(ValueError):
        rerank_results(
            candidates=[{"chunk_index": 1}],
            rerank_scores=[],
        )


def test_rerank_results_raises_error_when_top_k_is_not_positive():
    with pytest.raises(ValueError):
        rerank_results(
            candidates=[],
            rerank_scores=[],
            top_k=0,
        )
