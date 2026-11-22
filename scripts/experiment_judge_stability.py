# AI Judge의 신뢰성을 TEST

from app.generation_evaluation import (
    evaluate_faithfulness,
    calculate_faithfulness_score,
)


def main():
    context = """
        RAG uses DPR to retrieve passages.
        DPR follows a bi-encoder architecture.
    """

    answer = (
        "RAG uses Dense Passage Retriever (DPR), "
        "which follows a bi-encoder architecture."
    )

    scores = []

    from app.generation_evaluation import extract_claims, verify_claims

    claims = extract_claims(answer)

    print("\n--- Fixed Claims ---")
    for claim in claims:
        print(claim)

    for run in range(1, 6):
        result = verify_claims(
        context=context,
        claims=claims,
    )

        verdicts = [claim.supported for claim in result.claims]
        score = calculate_faithfulness_score(verdicts)
        scores.append(score)

        print(f"\n--- Run {run} ---")
        for claim in result.claims:
            print(f"Claim: {claim.claim}")
            print(f"Supported: {claim.supported}")
            print(f"Reason: {claim.reason}")
        print(f"Score: {score:.4f}")

    print("\n--- Stability Summary ---")
    print(f"Scores: {scores}")
    print(f"Mean: {sum(scores) / len(scores):.4f}")
    print(f"Range: {max(scores) - min(scores):.4f}")


if __name__ == "__main__":
    main()
