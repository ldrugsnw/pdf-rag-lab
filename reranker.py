def rerank_results(
    candidates: list[dict],
    rerank_scores: list[float],
    top_k: int = 3,
) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if len(candidates) != len(rerank_scores):
        raise ValueError(
            "candidates and rerank_scores must have the same length"
        )

    reranked_results = []

    for candidate, rerank_score in zip(candidates, rerank_scores):
        reranked_result = candidate.copy()
        reranked_result["rerank_score"] = rerank_score
        reranked_results.append(reranked_result)

    reranked_results.sort(
        key=lambda result: result["rerank_score"],
        reverse=True,
    )

    return reranked_results[:top_k]