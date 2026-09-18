import json
import sys
from pathlib import Path


CACHE_PATH = Path("cache/rag_paper_embeddings.json")


def parse_chunk_indices(arguments: list[str]) -> list[int]:
    if not arguments:
        raise SystemExit(
            "Usage: python -m scripts.inspect_chunks <chunk_index> [...]"
        )

    try:
        return [int(argument) for argument in arguments]
    except ValueError:
        raise SystemExit("chunk_index must be an integer")


def main():
    embedded_chunks = json.loads(
        CACHE_PATH.read_text(encoding="utf-8")
    )

    target_indices = parse_chunk_indices(sys.argv[1:])

    chunks_by_index = {
        chunk["chunk_index"]: chunk
        for chunk in embedded_chunks
    }

    for chunk_index in target_indices:
        chunk = chunks_by_index.get(chunk_index)

        if chunk is None:
            print(f"Chunk {chunk_index} not found")
            continue

        print("=" * 80)
        print(f'Chunk {chunk["chunk_index"]}')
        print(f'Pages: {chunk.get("page_numbers", [])}')
        print("-" * 80)
        print(chunk["text"])
        print()


if __name__ == "__main__":
    main()