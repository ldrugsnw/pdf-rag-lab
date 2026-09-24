import json
import sys
from collections import Counter
from pathlib import Path


def classify_outcome(
    retriever_rank: int | None,
    reranker_rank: int | None,
    k: int = 3,
) -> str:
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if retriever_rank is None:
        return "retrieval_failure"

    if reranker_rank is not None and reranker_rank <= k:
        return "reranker_recovered" if retriever_rank > k else "success"

    return (
        "reranker_regression"
        if retriever_rank <= k
        else "unresolved_ranking_failure"
    )


def classify_case(case: dict) -> list[str]:
    failure_types = []

    retriever_rank = case["retriever"]["first_relevant_rank"]
    reranker_rank = case["reranker"]["first_relevant_rank"]

    # Retriever Top 10 안에 정답이 없음
    if retriever_rank is None:
        failure_types.append("retrieval_failure")
    elif retriever_rank > 3:
        failure_types.append("retriever_top3_miss")

    # Reranker 이후에도 정답이 Top 3 밖에 있음
    if reranker_rank is None or reranker_rank > 3:
        failure_types.append("reranking_failure")

    # Reranker가 기존 정답 순위를 떨어뜨림
    if (
        retriever_rank is not None
        and reranker_rank is not None
        and reranker_rank > retriever_rank
    ):
        failure_types.append("reranking_regression")

    reranked_candidates = case["reranker"]["candidates"]

    relevant_scores = [
        candidate["rerank_score"]
        for candidate in reranked_candidates
        if candidate["is_relevant"]
    ]

    irrelevant_scores = [
        candidate["rerank_score"]
        for candidate in reranked_candidates
        if not candidate["is_relevant"]
    ]

    # 오답 청크가 정답과 같거나 더 높은 점수를 받음
    if relevant_scores and irrelevant_scores:
        highest_relevant_score = max(relevant_scores)
        highest_irrelevant_score = max(irrelevant_scores)

        if highest_irrelevant_score >= highest_relevant_score:
            failure_types.append("irrelevant_top_score")

    return failure_types


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m scripts.analyze_evaluation "
            "<evaluation_result_path>"
        )

    result_path = Path(sys.argv[1])

    if not result_path.exists():
        raise SystemExit(f"Result file not found: {result_path}")

    report = json.loads(result_path.read_text(encoding="utf-8"))

    analyzed_cases = []
    failure_counter = Counter()
    outcome_counter = Counter()

    for case in report["cases"]:
        failure_types = classify_case(case)
        outcome = classify_outcome(
            case["retriever"]["first_relevant_rank"],
            case["reranker"]["first_relevant_rank"],
        )

        analyzed_case = {
            **case,
            "outcome": outcome,
            "failure_types": failure_types,
        }

        analyzed_cases.append(analyzed_case)
        failure_counter.update(failure_types)
        outcome_counter.update([outcome])

    analyzed_report = {
        **report,
        "failure_summary": dict(failure_counter),
        "outcome_summary": dict(outcome_counter),
        "cases_without_flags": sum(
            1 for case in analyzed_cases if not case["failure_types"]
        ),
        "cases": analyzed_cases,
    }

    analysis_path = result_path.with_name(
        f"{result_path.stem}_analysis.json"
    )

    analysis_path.write_text(
        json.dumps(analyzed_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n--- Outcome Summary ---")
    for outcome, count in outcome_counter.items():
        print(f"{outcome}: {count}")

    print("\n--- Failure Summary ---")
    if failure_counter:
        for failure_type, count in failure_counter.items():
            print(f"{failure_type}: {count}")
    else:
        print("No failure flags detected.")

    print(
        "cases_without_flags: "
        f'{analyzed_report["cases_without_flags"]}'
    )
    print(f"\nAnalysis saved to: {analysis_path}")


if __name__ == "__main__":
    main()