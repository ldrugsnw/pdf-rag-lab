import json
from pathlib import Path


CACHE_PATH = Path("cache/rag_paper_embeddings.json")


def main():
    embedded_chunks = json.loads(
        CACHE_PATH.read_text(encoding="utf-8")
    )

    target_indices = {13, 27}

    for chunk in embedded_chunks:
        if chunk["chunk_index"] in target_indices:
            print("=" * 80)
            print(f'Chunk {chunk["chunk_index"]}')
            print(f'Pages: {chunk.get("page_numbers", [])}')
            print("-" * 80)
            print(chunk["text"])
            print()


if __name__ == "__main__":
    main()