
from app.generation_evaluation import evaluate_correctness


def main():
    question = "Which model does RAG use for retrieval?"
    reference_answer = "Dense Passage Retriever (DPR)."

    test_cases = [
        {
            "name": "Correct answer",
            "answer": "RAG uses DPR.",
        },
        {
            "name": "Correct core with incorrect detail",
            "answer": (
                "RAG uses DPR, which was developed "
                "by Google in 2022."
            ),
        },
        {
            "name": "Incorrect core answer",
            "answer": "RAG uses BM25 instead of DPR.",
        },
    ]

    for case in test_cases:
        result = evaluate_correctness(
            question=question,
            reference_answer=reference_answer,
            generated_answer=case["answer"],
        )

        print(f"\n--- {case['name']} ---")
        print(f"Answer: {case['answer']}")
        print(f"Correct: {result.correct}")
        print(f"Reason: {result.reason}")


if __name__ == "__main__":
    main()
