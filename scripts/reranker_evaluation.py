import json
from pathlib import Path

from app.embedder import embed_texts
from app.retrieval import search_chunks_by_embedding
from app.reranker import (
    calculate_rerank_scores_with_retry,
    rerank_results,
)
from datetime import datetime, timezone

CACHE_PATH = Path("cache/rag_paper_embeddings.json")
EVALUATION_PATH = Path("data/evaluation_cases.json")
RESULTS_DIR = Path("results")

def load_evaluation_cases() -> list[dict]:
    return json.loads(
        EVALUATION_PATH.read_text(encoding="utf-8")
    )


def find_first_relevant_rank(results, relevant_indices):
    for rank, result in enumerate(results, start=1):
        if result["chunk_index"] in relevant_indices:
            return rank

    return None


def reciprocal_rank(rank):
    if rank is None:
        return 0.0

    return 1.0 / rank


def hit_at_k(rank, k=3):
    if rank is None:
        return 0.0

    return 1.0 if rank <= k else 0.0


def main():
    embedded_chunks = json.loads(
        CACHE_PATH.read_text(encoding="utf-8")
    )

    retriever_hits = []
    retriever_rrs = []

    reranker_hits = []
    reranker_rrs = []

    evaluation_results = []

    evaluation_cases = load_evaluation_cases()

    for case in evaluation_cases:
        question = case["question"]
        relevant_indices = case["relevant_chunk_indices"]

        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print(f"Expected relevant chunks: {relevant_indices}")

        # Question embedding
        query_embedding = embed_texts([question])[0]

        # Retriever Top 10
        retrieved_chunks = search_chunks_by_embedding(
            query_embedding=query_embedding,
            chunks=embedded_chunks,
            top_k=10,
        )

        retriever_rank = find_first_relevant_rank(
            retrieved_chunks,
            relevant_indices,
        )

        # Real reranker
        rerank_scores = calculate_rerank_scores_with_retry(
            question=question,
            candidates=retrieved_chunks,
            max_retries=2
        )

        print("\nRetrieved chunks:")

        for rank, chunk in enumerate(retrieved_chunks, start=1):
            marker = (
                " <-- expected relevant"
                if chunk["chunk_index"] in relevant_indices
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
                if result["chunk_index"] in relevant_indices
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

        retriever_hit = hit_at_k(retriever_rank, k=3)
        retriever_rr = reciprocal_rank(retriever_rank)

        reranker_hit = hit_at_k(reranker_rank, k=3)
        reranker_rr = reciprocal_rank(reranker_rank)

        retriever_hits.append(retriever_hit)
        retriever_rrs.append(retriever_rr)

        reranker_hits.append(reranker_hit)
        reranker_rrs.append(reranker_rr)

        retrieved_candidates = []

        for rank, chunk in enumerate(retrieved_chunks, start=1):
            retrieved_candidates.append(
                {
                    "rank": rank,
                    "chunk_index": chunk["chunk_index"],
                    "retrieval_score": chunk["score"],
                    "is_relevant": (
                        chunk["chunk_index"] in relevant_indices
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
                        chunk["chunk_index"] in relevant_indices
                    ),
                }
            )

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
                    "candidates": retrieved_candidates,
                },
                "reranker": {
                    "first_relevant_rank": reranker_rank,
                    "hit_at_3": reranker_hit,
                    "reciprocal_rank": reranker_rr,
                    "candidates": reranked_candidates,
                },
            }
        )

        print(f"Retriever rank: {retriever_rank}")
        print(f"Reranker rank:  {reranker_rank}")
        print(
            f"Hit@3: {retriever_hit:.1f} -> "
            f"{reranker_hit:.1f}"
        )
        print(
            f"RR: {retriever_rr:.4f} -> "
            f"{reranker_rr:.4f}"
        )

    mean_retriever_hit = sum(retriever_hits) / len(retriever_hits)
    mean_reranker_hit = sum(reranker_hits) / len(reranker_hits)

    retriever_mrr = sum(retriever_rrs) / len(retriever_rrs)
    reranker_mrr = sum(reranker_rrs) / len(reranker_rrs)

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

    generated_at = datetime.now(timezone.utc)

    report = {
        "generated_at": generated_at.isoformat(),
        "number_of_cases": len(evaluation_results),
        "summary": {
            "retriever_mean_hit_at_3": mean_retriever_hit,
            "reranker_mean_hit_at_3": mean_reranker_hit,
            "retriever_mrr": retriever_mrr,
            "reranker_mrr": reranker_mrr,
        },
        "cases": evaluation_results,
    }

    RESULTS_DIR.mkdir(exist_ok=True)

    timestamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    result_path = RESULTS_DIR / f"reranker_evaluation_{timestamp}.json"

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

    