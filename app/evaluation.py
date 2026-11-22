# Retrieval 평가 지표들임


def calculate_reciprocal_rank(
    results: list[dict],
    relevant_chunk_indices: list[int],
) -> float:
    
    relevant_indices = set(relevant_chunk_indices)

    for rank, result in enumerate(results, start=1):
        if result["chunk_index"] in relevant_indices:
            return 1.0 / rank
        
    return 0.0


def calculate_hit_at_k(
    results: list[dict],
    relevant_chunk_indices: list[int],
    k: int,
) -> float:
    
    if k <= 0:
        raise ValueError("k must be greater than 0")

    relevant_indices = set(relevant_chunk_indices)

    for result in results[:k]:
        if result["chunk_index"] in relevant_indices:
            return 1.0

    return 0.0

def calculate_recall_at_k(
    results: list[dict],
    relevant_chunk_indices: list[int],
    k: int,
) -> float : 
    
    if k <= 0:
        raise ValueError("k must be greater than 0")

    relevant_indices = set(relevant_chunk_indices)
    
    found = set()

    for result in results[:k]:
        if result["chunk_index"] in relevant_indices:
            found.add(result["chunk_index"])

    if not relevant_indices:
        return 0.0

    return len(found) / len(relevant_indices)
