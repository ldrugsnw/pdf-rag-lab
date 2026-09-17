import json
from pathlib import Path

from app.embedder import embed_texts
from app.retrieval import search_chunks_by_embedding
from app.reranker import (
    RerankingError,
    calculate_rerank_scores_with_retry,
    rerank_results,
)
from app.generator import generate_answer


CACHE_PATH = Path("cache/rag_paper_embeddings.json")

embedded_chunks = json.loads(
    CACHE_PATH.read_text(encoding="utf-8")
)

question = "Which model does RAG use to retrieve passages from its document index?"
# question = "Who was the first person to walk on the Moon?"

# 1. Embed question
query_embedding = embed_texts([question])[0]

# 2. Retrieve candidates
retrieved_chunks = search_chunks_by_embedding(
    query_embedding=query_embedding,
    chunks=embedded_chunks,
    top_k=10,
)

# 3. Rerank candidates
try:
    rerank_scores = calculate_rerank_scores_with_retry(
        question=question,
        candidates=retrieved_chunks,
        max_retries=2,
    )

    selected_chunks = rerank_results(
        candidates=retrieved_chunks,
        rerank_scores=rerank_scores,
        top_k=3,
    )

except RerankingError:
    print(
        "Reranking failed. "
        "Falling back to retriever results."
    )

    selected_chunks = retrieved_chunks[:3]

print("임시 로그\n")

print("\n--- Retriever Top 10 ---")
for rank, chunk in enumerate(retrieved_chunks, start=1):
    marker = " <-- relevant" if chunk["chunk_index"] == 13 else ""
    print(
        f'{rank}. chunk={chunk["chunk_index"]}, '
        f'score={chunk["score"]:.4f}{marker}'
    )

print("\n--- Selected After Reranking ---")
for rank, chunk in enumerate(selected_chunks, start=1):
    marker = " <-- relevant" if chunk["chunk_index"] == 13 else ""
    print(
        f'{rank}. chunk={chunk["chunk_index"]}, '
        f'rerank={chunk.get("rerank_score", "fallback")}'
        f'{marker}'
    )
# 4. Generate answer
answer = generate_answer(
    question=question,
    chunks=selected_chunks,
)

print("\n--- Question ---")
print(question)

print("\n--- Final Context ---")
for chunk in selected_chunks:
    print(
        f'chunk={chunk["chunk_index"]}, '
        f'pages={chunk["page_numbers"]}'
    )

print("\n--- Answer ---")
print(answer)