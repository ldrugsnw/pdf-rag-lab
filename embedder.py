from openai import OpenAI


MODEL_NAME = "text-embedding-3-small"

client = OpenAI()


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("texts must not contain empty strings")

    response = client.embeddings.create(
        model=MODEL_NAME,
        input=texts,
    )

    ordered_data = sorted(
        response.data,
        key=lambda item: item.index,
    )

    return [item.embedding for item in ordered_data]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunk = chunk.copy()

        # embedded_chunk에 "embedding" 키 추가
        embedded_chunk["embedding"] = embedding

        # embedded_chunks 리스트에 추가
        embedded_chunks.append(embedded_chunk)

    return embedded_chunks