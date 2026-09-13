import math

def calculate_keyword_score(query: str, chunk_text: str) -> float:
    query_words = set(query.lower().split())
    chunk_words = set(chunk_text.lower().split())

    if not query_words:
        return 0.0

    common_words = query_words & chunk_words

    score = len(common_words) / len(query_words)

    return score


def search_chunks(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    results = []

    for chunk in chunks:
        score = calculate_keyword_score(query, chunk["text"])

        if score > 0:
            results.append(
                {
                    "score": score,
                    "chunk_index": chunk["chunk_index"],
                    "page_numbers": chunk["page_numbers"],
                    "text": chunk["text"],
                }
            )

    results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    return results[:top_k]


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    if len(vector_a) != len(vector_b):
        raise ValueError("vectors must have the same dimension")

    dot_product = 0.0

    for value_a, value_b in zip(vector_a, vector_b):
        dot_product += value_a * value_b

    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))

    if norm_a == 0 or norm_b == 0:
        raise ValueError("vectors must not be zero vectors")

    return dot_product / (norm_a * norm_b)
