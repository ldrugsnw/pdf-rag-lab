from app.generator import build_prompt


def test_build_prompt():
    chunks = [
        {
            "chunk_index": 3,
            "page_numbers": [2, 3],
            "text": "RAG combines retrieval with generation.",
        },
        {
            "chunk_index": 7,
            "page_numbers": [5],
            "text": "Retrieved documents are provided to the generator.",
        },
    ]

    prompt = build_prompt("What is RAG?", chunks)

    assert "What is RAG?" in prompt
    assert "RAG combines retrieval with generation." in prompt
    assert "Retrieved documents are provided to the generator." in prompt
    assert "Pages 2, 3" in prompt





def test_build_prompt_when_multiple_pages():
    chunks = [
        {
            "chunk_index": 10,
            "page_numbers": [3, 4],
            "text": "Example context.",
        }
    ]

    prompt = build_prompt("Example question?", chunks)

    assert "Pages 3, 4" in prompt
    assert "Example context." in prompt
    assert "Example question?" in prompt