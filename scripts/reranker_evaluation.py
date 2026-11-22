
import json
from datetime import datetime, timezone
from pathlib import Path

from app.embedder import embed_texts
from app.evaluation import calculate_recall_at_k
from app.retrieval import search_chunks_by_embedding
from app.reranker import (
    calculate_rerank_scores_with_retry,
    rerank_results,
)

CACHE_PATH = Path("cache/rag_paper_embeddings_v2.json")
EVALUATION_PATH = Path("data/evaluation_cases.json")
RESULTS_DIR = Path("results")


def load_evaluation_cases() -> list[dict]:
    return json.loads(
        EVALUATION_PATH.read_text(encoding="utf-8")
    )


def find_first_relevant_rank(
    results: list[dict],
    relevant_indices: list[int],
) -> int | None:
    relevant_set = set(relevant_indices)

    for rank, result in enumerate(results, start=1):
        if result["chunk_index"] in relevant_set:
            return rank

    return None


def reciprocal_rank(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / rank


def hit_at_k(rank: int | None, k: int = 3) -> float:
    if rank is None:
        return 0.0
    return 1.0 if rank <= k else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main():
    print("Loading chunk embeddings from cache...")

    embedded_chunks = json.loads(
        CACHE_PATH.read_text(encoding="utf-8")
    )

    evaluation_cases = load_evaluation_cases()

    retriever_hits = []
    retriever_rrs = []
    retriever_recalls = []

    reranker_hits = []
    reranker_rrs = []
    reranker_recalls = []

    evaluation_results = []

    for case in evaluation_cases:
        question = case["question"]
        relevant_indices = case["relevant_chunk_indices"]
        relevant_set = set(relevant_indices)

        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print(f"Expected relevant chunks: {relevant_indices}")

        # 1. Embed question
        query_embedding = embed_texts([question])[0]

        # 2. Retrieve top 10 candidates
        retrieved_chunks = search_chunks_by_embedding(
            query_embedding=query_embedding,
            chunks=embedded_chunks,
            top_k=10,
        )

        retriever_rank = find_first_relevant_rank(
            retrieved_chunks,
            relevant_indices,
        )

        # 3. Rerank the retrieved candidates
        rerank_scores = calculate_rerank_scores_with_retry(
            question=question,
            candidates=retrieved_chunks,
            max_retries=2,
        )

        print("\nRetrieved chunks:")

        for rank, chunk in enumerate(retrieved_chunks, start=1):
            marker = (
                " <-- expected relevant"
                if chunk["chunk_index"] in relevant_set
                else ""
            )

            print(
                f'{rank}. chunk={chunk["chunk_index"]}, '
                f'retrieval={chunk["score"]:.4f}'
                f'{marker}'
            )

        print("\nRaw rerank scores:")

        for result in rerank_scores:
            marker = (
                " <-- expected relevant"
                if result["chunk_index"] in relevant_set
                else ""
            )

            print(
                f'chunk={result["chunk_index"]}, '
                f'rerank={result["score"]:.4f}'
                f'{marker}'
            )

        reranked_chunks = rerank_results(
            candidates=retrieved_chunks,
            rerank_scores=rerank_scores,
            top_k=10,
        )

        reranker_rank = find_first_relevant_rank(
            reranked_chunks,
            relevant_indices,
        )

        # 4. Retriever metrics
        retriever_hit = hit_at_k(retriever_rank, k=3)
        retriever_rr = reciprocal_rank(retriever_rank)
        retriever_recall = calculate_recall_at_k(
            results=retrieved_chunks,
            relevant_chunk_indices=relevant_indices,
            k=3,
        )

        # 5. Reranker metrics
        reranker_hit = hit_at_k(reranker_rank, k=3)
        reranker_rr = reciprocal_rank(reranker_rank)
        reranker_recall = calculate_recall_at_k(
            results=reranked_chunks,
            relevant_chunk_indices=relevant_indices,
            k=3,
        )

        # 6. Collect metrics
        retriever_hits.append(retriever_hit)
        retriever_rrs.append(retriever_rr)
        retriever_recalls.append(retriever_recall)

        reranker_hits.append(reranker_hit)
        reranker_rrs.append(reranker_rr)
        reranker_recalls.append(reranker_recall)

        # 7. Build candidate records
        retrieved_candidates = []

        for rank, chunk in enumerate(retrieved_chunks, start=1):
            retrieved_candidates.append(
                {
                    "rank": rank,
                    "chunk_index": chunk["chunk_index"],
                    "retrieval_score": chunk["score"],
                    "is_relevant": (
                        chunk["chunk_index"] in relevant_set
                    ),
                }
            )

        reranked_candidates = []

        for rank, chunk in enumerate(reranked_chunks, start=1):
            reranked_candidates.append(
                {
                    "rank": rank,
                    "chunk_index": chunk["chunk_index"],
                    "retrieval_score": chunk["score"],
                    "rerank_score": chunk["rerank_score"],
                    "is_relevant": (
                        chunk["chunk_index"] in relevant_set
                    ),
                }
            )

        # 8. Save per-question results
        evaluation_results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "query_style": case.get("query_style"),
                "question": question,
                "relevant_chunk_indices": relevant_indices,
                "retriever": {
                    "first_relevant_rank": retriever_rank,
                    "hit_at_3": retriever_hit,
                    "reciprocal_rank": retriever_rr,
                    "recall_at_3": retriever_recall,
                    "candidates": retrieved_candidates,
                },
                "reranker": {
                    "first_relevant_rank": reranker_rank,
                    "hit_at_3": reranker_hit,
                    "reciprocal_rank": reranker_rr,
                    "recall_at_3": reranker_recall,
                    "candidates": reranked_candidates,
                },
            }
        )

        # 9. Print per-question comparison
        print(f"\nRetriever rank: {retriever_rank}")
        print(f"Reranker rank:  {reranker_rank}")

        print(
            f"Hit@3: {retriever_hit:.4f} -> "
            f"{reranker_hit:.4f}"
        )
        print(
            f"RR: {retriever_rr:.4f} -> "
            f"{reranker_rr:.4f}"
        )
        print(
            f"Recall@3: {retriever_recall:.4f} -> "
            f"{reranker_recall:.4f}"
        )

    # 10. Calculate overall metrics
    mean_retriever_hit = mean(retriever_hits)
    mean_reranker_hit = mean(reranker_hits)

    retriever_mrr = mean(retriever_rrs)
    reranker_mrr = mean(reranker_rrs)

    mean_retriever_recall = mean(retriever_recalls)
    mean_reranker_recall = mean(reranker_recalls)

    print("\n--- Evaluation Summary ---")

    print(
        f"Mean Hit@3: "
        f"{mean_retriever_hit:.4f} -> "
        f"{mean_reranker_hit:.4f}"
    )
    print(
        f"MRR: "
        f"{retriever_mrr:.4f} -> "
        f"{reranker_mrr:.4f}"
    )
    print(
        f"Mean Recall@3: "
        f"{mean_retriever_recall:.4f} -> "
        f"{mean_reranker_recall:.4f}"
    )

    # 11. Save JSON report
    generated_at = datetime.now(timezone.utc)

    report = {
        "generated_at": generated_at.isoformat(),
        "number_of_cases": len(evaluation_results),
        "summary": {
            "retriever_mean_hit_at_3": mean_retriever_hit,
            "reranker_mean_hit_at_3": mean_reranker_hit,
            "retriever_mrr": retriever_mrr,
            "reranker_mrr": reranker_mrr,
            "retriever_mean_recall_at_3": mean_retriever_recall,
            "reranker_mean_recall_at_3": mean_reranker_recall,
        },
        "cases": evaluation_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    result_path = (
        RESULTS_DIR / f"reranker_evaluation_{timestamp}.json"
    )

    result_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nEvaluation report saved to: {result_path}")


if __name__ == "__main__":
    main()
