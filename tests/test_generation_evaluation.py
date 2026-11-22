from app.generation_evaluation import calculate_faithfulness_score
from app.generation_evaluation import build_correctness_prompt

from unittest.mock import patch

from app.generation_evaluation import (
    ClaimVerdict,
    FaithfulnessResult,
    calculate_answer_faithfulness,
)

def test_faithfulness_score():
    assert calculate_faithfulness_score([True, False, True, True]) == 0.75
    assert calculate_faithfulness_score([True, True]) == 1.0
    assert calculate_faithfulness_score([False, False]) == 0.0
    assert calculate_faithfulness_score([]) == 0.0



def test_calculate_answer_faithfulness():
    fake_result = FaithfulnessResult(
        claims=[
            ClaimVerdict(
                claim="RAG uses DPR.", 
                supported=True,
                reason="The context explicitly states that RAG uses DPR.",)
            ,
            ClaimVerdict(
                claim="DPR is faster than BM25.",
                supported=False,
                reason="The context explicitly states that RAG uses DPR.",
            ),
        ]
    )

    with patch(
        "app.generation_evaluation.evaluate_faithfulness",
        return_value=fake_result,
    ) as mock_judge:
        score = calculate_answer_faithfulness(
            context="RAG uses DPR.",
            answer="RAG uses DPR. DPR is faster than BM25.",
        )

    assert score == 0.5
    mock_judge.assert_called_once()


def test_build_correctness_prompt():
    prompt = build_correctness_prompt(
        question="Which model does RAG use for retrieval?",
        reference_answer="Dense Passage Retriever (DPR).",
        generated_answer="RAG uses DPR.",
    )

    assert "Which model does RAG use for retrieval?" in prompt
    assert "Dense Passage Retriever (DPR)." in prompt
    assert "RAG uses DPR." in prompt
