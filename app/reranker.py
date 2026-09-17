import json

from openai import OpenAI

client = OpenAI()

class RerankingError(Exception):
    """Raised when reranking fails after retries."""

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {
                        "type": "string"
                    },
                    "score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["candidate_id", "score"],
                "additionalProperties": False
            }
        }
    },
    "required": ["scores"],
    "additionalProperties": False
}


def build_rerank_prompt(question: str, candidates: list[dict]) -> str:

    instruction = """
    Given a question and candidate passages, evaluate how useful
    each candidate is for answering the question.

    Score every candidate from 0.0 to 1.0.

    Scoring guideline:
    - 0.0: Completely irrelevant to answering the question.
    - 0.5: Related to the question, but does not directly answer it.
    - 1.0: Contains key information that directly answers the question.

    Important:
    - Do not give a high score merely because a candidate contains
    keywords or terms related to the question.
    - Give a high score only when the candidate's context provides
    evidence that directly supports an answer to the question.
    - Evaluate every candidate exactly once.
    - Return only the candidate_id shown in the candidate header.
    """

    candidate_texts = []


    for position, candidate in enumerate(candidates):
        candidate_text = f"""
    CANDIDATE_{position}:
    {candidate["text"]}
    """
        candidate_texts.append(candidate_text)

    candidates_section = "\n".join(candidate_texts)

    return f"""
        {instruction}

        Question:
        {question}

        Candidates:
        {candidates_section}
        """.strip()

def calculate_rerank_scores(
    question: str,
    candidates: list[dict],
) -> list[dict]:
    prompt = build_rerank_prompt(question, candidates)

    #print("\n--- ACTUAL RERANK PROMPT ---")
    #print(prompt)
    #print("--- END PROMPT ---\n")

    response = client.responses.create(
        model = "gpt-5-nano",
        input = prompt,
        text = {
            "format": {
                 "type": "json_schema",
                 "name": "rerank_scores",
                 "strict": True,
                 "schema": RERANK_SCHEMA,
            }
        },
    )

    result = json.loads(response.output_text)
    scores = result["scores"]

    candidate_map = {
    f"CANDIDATE_{position}": candidate["chunk_index"]
    for position, candidate in enumerate(candidates)
    }

    rerank_scores = []

    for item in scores:
        candidate_id = item["candidate_id"]
        if candidate_id not in candidate_map:
            raise ValueError(
                f"Unexpected candidate_id: {candidate_id}"
            )

        rerank_scores.append(
            {
                "chunk_index": candidate_map[candidate_id],
                "score": item["score"],
            }
        )

    validate_rerank_scores(
        candidates=candidates,
        rerank_scores=rerank_scores,
    )

    return rerank_scores

def calculate_rerank_scores_with_retry(
    question: str,
    candidates: list[dict],
    max_retries: int = 2,
) -> list[dict]:
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            scores = calculate_rerank_scores(
                question=question,
                candidates=candidates,
            )

            print(
                f"Reranking succeeded "
                f"(attempt {attempt + 1}/{max_retries + 1})"
            )

            return scores

        except (ValueError, json.JSONDecodeError) as error:
            last_error = error

            print(
                f"Reranking attempt "
                f"{attempt + 1}/{max_retries + 1} failed: "
                f"{error}"
            )

    raise RerankingError(
        f"Reranking failed after {max_retries + 1} attempts"
    ) from last_error



def validate_rerank_scores(
    candidates: list[dict],
    rerank_scores: list[dict],
) -> None:
    expected_indices = {
        candidate["chunk_index"]
        for candidate in candidates
    }

    returned_indices = [
        result["chunk_index"]
        for result in rerank_scores
    ]

    if len(returned_indices) != len(set(returned_indices)):
        raise ValueError("Duplicate chunk_index in rerank scores")

    returned_index_set = set(returned_indices)

    if returned_index_set != expected_indices:
        missing = expected_indices - returned_index_set
        unexpected = returned_index_set - expected_indices

        raise ValueError(
            f"Invalid rerank scores: "
            f"missing={missing}, unexpected={unexpected}"
        )

def rerank_results(
    candidates: list[dict],
    rerank_scores: list[dict],
    top_k: int = 3,
) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    score_by_chunk_index = {
        result["chunk_index"]: result["score"]
        for result in rerank_scores
    }

    reranked_results = []

    for candidate in candidates:
        chunk_index = candidate["chunk_index"]

        if chunk_index not in score_by_chunk_index:
            raise ValueError(
                f"Missing rerank score for chunk {chunk_index}"
            )

        reranked_result = candidate.copy()
        reranked_result["rerank_score"] = score_by_chunk_index[chunk_index]

        reranked_results.append(reranked_result)

    reranked_results.sort(
        key=lambda result: result["rerank_score"],
        reverse=True,
    )

    return reranked_results[:top_k]