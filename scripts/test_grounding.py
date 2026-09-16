from app.generator import generate_answer


def main():
    question = "What is the capital of France?"

    fake_chunks = [
        {
            "chunk_index": 0,
            "page_numbers": [1],
            "text": "The capital of France is Lyon.",
        }
    ]

    answer = generate_answer(
        question=question,
        chunks=fake_chunks,
    )

    print(answer)


if __name__ == "__main__":
    main()
