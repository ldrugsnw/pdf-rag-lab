import json
from pathlib import Path

from app.embedder import embed_texts
from app.retrieval import search_chunks_by_embedding
# from app.reranker import rerank_chunks
from app.generator import generate_answer
from app.reranker import rerank_results


CACHE_PATH = Path("cache/rag_paper_embeddings.json")

embedded_chunks = json.loads(
    CACHE_PATH.read_text(encoding="utf-8")
)

question = "Who was the first person to walk on the Moon?"

# 1. Embed question
query_embedding = embed_texts([question])[0]

# 2. Retrieve candidates
retrieved_chunks = search_chunks_by_embedding(
    query_embedding=query_embedding,
    chunks=embedded_chunks,
    top_k=10,
)

# 3. Rerank candidates
# 3. Fake reranker

fake_scores = [

    1.0 - (i * 0.05)

    for i in range(len(retrieved_chunks))

]

reranked_chunks = rerank_results(

    candidates=retrieved_chunks,

    rerank_scores=fake_scores,

    top_k=3,
)

# 4. Select final context
context_chunks = reranked_chunks[:3]

# 5. Generate answer
answer = generate_answer(
    question=question,
    chunks=context_chunks,
)

print("\n--- Question ---")
print(question)

print("\n--- Retrieved Context ---")
for chunk in context_chunks:
    print(
        f'chunk={chunk["chunk_index"]}, '
        f'pages={chunk["page_numbers"]}'
    )

print("\n--- Answer ---")
print(answer)