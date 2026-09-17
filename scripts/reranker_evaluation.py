import json
from pathlib import Path

from app.embedder import embed_texts
from app.retrieval import search_chunks_by_embedding
from app.reranker import (
    calculate_rerank_scores_with_retry,
    rerank_results,
)

CACHE_PATH = Path("cache/rag_paper_embeddings.json")


EVALUATION_CASES = [
    {
        "question": (
            "Which model does RAG use to retrieve passages "
            "from its document index?"
        ),
        "relevant_chunk_indices": [13],
    },
    {
        "question": (
            "What is the difference between RAG-Sequence "
            "and RAG-Token?"
        ),
        "relevant_chunk_indices": [11],
    },
    {
        "question": (
            "Why do the authors keep the passage encoder "
            "fixed during training?"
        ),
        "relevant_chunk_indices": [15],
    },
]


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

    for case in EVALUATION_CASES:
        question = case["question"]
        relevant_indices = case["relevant_chunk_indices"]

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
            print(
                f'{rank}. chunk={chunk["chunk_index"]}, '
                f'retrieval={chunk["score"]:.4f}'
            )

        print("\nRaw rerank scores:")

        for result in rerank_scores:
            print(
                f'chunk={result["chunk_index"]}, '
                f'rerank={result["score"]:.4f}'
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

        print("=" * 60)
        print(f"Question: {question}")
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

    


if __name__ == "__main__":
    main()

    