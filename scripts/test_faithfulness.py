
from app.generation_evaluation import (
    evaluate_faithfulness,
    calculate_faithfulness_score,
)


def main():
    context = "RAG uses DPR to retrieve documents."

    answer = (
        "RAG uses DPR to retrieve documents. "
        "DPR is faster than BM25."
    )

    result = evaluate_faithfulness(
        context=context,
        answer=answer,
    )

    print("\n--- Claim Verdicts ---")

    for claim in result.claims:
        print(f"Claim: {claim.claim}")
        print(f"Supported: {claim.supported}")
        print()

    score = calculate_faithfulness_score(
        [claim.supported for claim in result.claims]
    )

    print(f"Faithfulness Score: {score:.4f}")


if __name__ == "__main__":
    main()
