from app import embedder


def test_embed_chunks_adds_embeddings_without_changing_original(monkeypatch):
    chunks = [
        {
            "chunk_index": 0,
            "page_numbers": [1],
            "text": "first chunk",
        },
        {
            "chunk_index": 1,
            "page_numbers": [2],
            "text": "second chunk",
        },
    ]

    def fake_embed_texts(texts):
        assert texts == ["first chunk", "second chunk"]

        return [
            [1.0, 0.0],
            [0.0, 1.0],
        ]

    monkeypatch.setattr(
        embedder,
        "embed_texts",
        fake_embed_texts,
    )

    result = embedder.embed_chunks(chunks)

    assert result[0]["text"] == "first chunk"
    assert result[0]["chunk_index"] == 0
    assert result[0]["embedding"] == [1.0, 0.0]

    assert result[1]["text"] == "second chunk"
    assert result[1]["chunk_index"] == 1
    assert result[1]["embedding"] == [0.0, 1.0]

    assert "embedding" not in chunks[0]
    assert "embedding" not in chunks[1]
