import pytest
from app.retrieval import (
    cosine_similarity,
    search_chunks,
    search_chunks_by_embedding,
)

def test_search_chunks_returns_highest_scoring_chunk():
    chunks = [
        {
            "chunk_index": 0,
            "page_numbers": [1],
            "text": "python web framework",
        },
        {
            "chunk_index": 1,
            "page_numbers": [2],
            "text": "python machine learning model",
        },
        {
            "chunk_index": 2,
            "page_numbers": [3],
            "text": "database query language",
        }
    ]

    results = search_chunks(
        query="python machine learning",
        chunks=chunks,
        top_k=1,
    )

    assert len(results)==1
    assert results[0]["chunk_index"]==1
    assert results[0]["score"]==1.0
    assert results[0]["page_numbers"]==[2]


def test_search_chunks_returns_empty_list_when_no_chunks_match():    
    chunks=[
        {
            "chunk_index": 0,
            "page_numbers": [1],
            "text": "python web framework",
        }
    ]

    results = search_chunks(
        query="cpp",
        chunks=chunks,
    )

    assert results == []

def test_search_chunks_returns_empty_list_when_query_is_empty():
    chunks=[
            {
                "chunk_index": 0,
                "page_numbers": [1],
                "text": "python web framework",
            }
        ]

    results = search_chunks(
        query="",
        chunks=chunks,
    )

    assert results == []

def test_search_chunks_raises_error_when_top_k_is_not_positive():
    with pytest.raises(ValueError):
        search_chunks(
            query="python",
            chunks=[],
            top_k=0,
        )


def test_cosine_similarity_returns_one_for_same_direction():
    similarity = cosine_similarity(
        vector_a=[1.0, 2.0],
        vector_b=[10.0, 20.0],
    )

    assert similarity == pytest.approx(1.0) # 근삿값으로 계산

def test_cosine_similarity_returns_zero_for_perpendicular_vectors():
    similarity = cosine_similarity(
        vector_a=[1.0, 0.0],
        vector_b=[0.0, 1.0],
    )

    assert similarity == pytest.approx(0.0)

def test_cosine_similarity_raises_valueError_for_different_dimension_vectors():
    vector_a=[1.0, 2.0]
    vector_b=[1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        cosine_similarity(
            vector_a,
            vector_b,
        )

def test_cosine_similarity_raises_valueError_for_zero_vector():
    vector_a=[0.0, 0.0]
    vector_b=[1.0, 3.0]

    with pytest.raises(ValueError):
        cosine_similarity(
            vector_a,
            vector_b,
        )

def test_search_chunks_by_embedding_returns_most_similar_chunk():
    query_embedding = [1.0, 0.0]

    chunks = [
        {
            "chunk_index": 1,
            "page_numbers": [2],
            "text": "수직 방향",
            "embedding": [0.0, 1.0],
        },
        {
            "chunk_index": 0,
            "page_numbers": [1],
            "text": "같은 방향",
            "embedding": [1.0, 0.0],
        },
    ]

    results = search_chunks_by_embedding(
        query_embedding=query_embedding,
        chunks=chunks,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["chunk_index"] == 0
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[0]["page_numbers"] == [1]
