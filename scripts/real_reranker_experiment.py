import json
from pathlib import Path

from app.embedder import embed_texts
from app.retrieval import search_chunks_by_embedding
from app.reranker import calculate_rerank_scores, rerank_results


CACHE_PATH = Path("cache/rag_paper_embeddings.json")


def main():
    embedded_chunks = json.loads(
        CACHE_PATH.read_text(encoding="utf-8")
    )

    question = (
    "Which model does RAG use to retrieve passages "
    "from its document index?"
    )

    relevant_chunk_index = 13

    # relevant_chunk_index = 15

    # 1. Question embedding
    query_embedding = embed_texts([question])[0]

    # 2. Retriever Top 10
    retrieved_chunks = search_chunks_by_embedding(
        query_embedding=query_embedding,
        chunks=embedded_chunks,
        top_k=10,
    )

    print("\n--- Retriever Top 10 ---")

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        marker = (
            " <-- relevant"
            if chunk["chunk_index"] == relevant_chunk_index
            else ""
        )

        print(
            f'{rank}. chunk={chunk["chunk_index"]}, '
            f'score={chunk["score"]:.4f}'
            f'{marker}'
        )

    # 3. Real reranker
    rerank_scores = calculate_rerank_scores(
        question=question,
        candidates=retrieved_chunks,
    )

    print("\n--- Raw Rerank Scores ---")
    print(rerank_scores)

    # 4. Rerank
    reranked_chunks = rerank_results(
        candidates=retrieved_chunks,
        rerank_scores=rerank_scores,
        top_k=10,
    )

    print("\n--- Real Reranker Top 10 ---")

    for rank, chunk in enumerate(reranked_chunks, start=1):
        marker = (
            " <-- relevant"
            if chunk["chunk_index"] == relevant_chunk_index
            else ""
        )

        print(
            f'{rank}. chunk={chunk["chunk_index"]}, '
            f'retrieval={chunk["score"]:.4f}, '
            f'rerank={chunk["rerank_score"]:.4f}'
            f'{marker}'
        )


if __name__ == "__main__":
    main()