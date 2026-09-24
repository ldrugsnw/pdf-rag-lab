import pytest

from scripts.analyze_evaluation import classify_outcome


@pytest.mark.parametrize(
    ("retriever_rank", "reranker_rank", "expected"),
    [
        (None, None, "retrieval_failure"),
        (4, 3, "reranker_recovered"),
        (3, 4, "reranker_regression"),
        (4, 4, "unresolved_ranking_failure"),
        (3, 3, "success"),
        (1, 2, "success"),
    ],
)
def test_classify_outcome(
    retriever_rank,
    reranker_rank,
    expected,
):
    outcome = classify_outcome(
        retriever_rank=retriever_rank,
        reranker_rank=reranker_rank,
    )

    assert outcome == expected


def test_classify_outcome_rejects_invalid_k():
    with pytest.raises(ValueError):
        classify_outcome(
            retriever_rank=1,
            reranker_rank=1,
            k=0,
        )