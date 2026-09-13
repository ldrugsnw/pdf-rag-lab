import json

from pathlib import Path

from app.chunker import chunk_pages
from app.parser import extract_pages
from app.embedder import embed_chunks, embed_texts
from app.retrieval import search_chunks_by_embedding

from app.evaluation import (
    calculate_hit_at_k,
    calculate_reciprocal_rank,
)

PDF_PATH = Path("samples/rag_paper.pdf")
CACHE_PATH = Path("cache/rag_paper_embeddings.json")

pdf_bytes = PDF_PATH.read_bytes()
pages = extract_pages(pdf_bytes)
chunks = chunk_pages(pages)

print("pages:", len(pages))
print("chunks:", len(chunks))

"""
for chunk in chunks[:3]:
    print("=" * 50)
    print("chunk_index:", chunk["chunk_index"])
    print("page_numbers:", chunk["page_numbers"])
    print(chunk["text"][:500])
"""

#################################################################

if CACHE_PATH.exists():
    print("Loading chunk embeddings from cache...")

    embedded_chunks = json.loads(
        CACHE_PATH.read_text(encoding="utf-8")
    )

else:
    print("Creating chunk embeddings...")

    embedded_chunks = embed_chunks(chunks)

    CACHE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CACHE_PATH.write_text(
        json.dumps(embedded_chunks, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Saved chunk embeddings to cache.")

    
evaluation_cases = [
    {
        "query": (
            "Which model does RAG use to retrieve passages "
            "from its document index?"
        ),
        "relevant_chunk_indices": [13],
    },
    {
        "query": (
            "What is the difference between "
            "RAG-Sequence and RAG-Token?"
        ),
        "relevant_chunk_indices": [11],
    },
    {
        "query": (
            "Why do the authors keep the passage encoder "
            "fixed during training?"
        ),
        "relevant_chunk_indices": [15],
    },
]

queries = [
    case["query"]
    for case in evaluation_cases
]

query_embeddings = embed_texts(queries)

hit_scores = []
reciprocal_ranks = []

for case, query_embedding in zip(
    evaluation_cases,
    query_embeddings,
):
    results = search_chunks_by_embedding(
        query_embedding=query_embedding,
        chunks=embedded_chunks,
        top_k=len(embedded_chunks),
    )

    hit_at_3 = calculate_hit_at_k(
        results=results,
        relevant_chunk_indices=case["relevant_chunk_indices"],
        k=3,
    )

    reciprocal_rank = calculate_reciprocal_rank(
        results=results,
        relevant_chunk_indices=case["relevant_chunk_indices"],
    )

    hit_scores.append(hit_at_3)
    reciprocal_ranks.append(reciprocal_rank)

    if reciprocal_rank > 0:
        first_relevant_rank = round(1 / reciprocal_rank)
    else:
        first_relevant_rank = None

    print("=" * 60)
    print("query:", case["query"])
    print("first relevant rank:", first_relevant_rank)
    print("Hit@3:", hit_at_3)
    print("RR:", round(reciprocal_rank, 4))


mean_hit_at_3 = sum(hit_scores) / len(hit_scores)
mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

print("\n--- Evaluation Summary ---")
print("Mean Hit@3:", round(mean_hit_at_3, 4))
print("MRR:", round(mrr, 4))
