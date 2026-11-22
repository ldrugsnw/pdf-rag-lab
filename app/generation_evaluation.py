from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

class ClaimVerdict(BaseModel):
    claim: str
    supported: bool
    reason: str

class CorrectnessResult(BaseModel):
    correct: bool
    reason: str



class FaithfulnessResult(BaseModel):
    claims: list[ClaimVerdict]


def calculate_faithfulness_score(
    claim_verdicts: list[bool],
) -> float:
    # TODO 1: 빈 리스트면 0.0 반환

    if len(claim_verdicts) == 0:
        return 0.0
    
    # TODO 2: True인 Claim 개수 계산
    supported_count = sum(claim_verdicts)

    # TODO 3: 전체 Claim 개수로 나누기
    return supported_count / len(claim_verdicts)


def build_faithfulness_prompt(
    context: str,
    answer: str,
) -> str:
    prompt = f"""
You are a faithfulness evaluator for a RAG system.

Your task:
1. Extract factual claims from the generated answer.
2. Determine whether each claim is supported by the context.

Rules:
- Use ONLY the provided context.
- Do NOT use external knowledge.
- Mark a claim as supported only if the context
  directly states or logically entails it.
- If the context does not support a claim,
  mark it as unsupported.
- Do not treat the absence of evidence as proof
  that a claim is factually false.
- For each claim, provide a brief reason explaining
  why it is supported or unsupported.
- When supported, cite the relevant evidence from
  the provided context in your reason.
- When unsupported, explain what evidence is missing.

Context:
{context}

Generated Answer:
{answer}
"""
    return prompt


def evaluate_faithfulness(
    context: str,
    answer: str,
) -> FaithfulnessResult:

    prompt = build_faithfulness_prompt(
        context=context,
        answer=answer,
    )

    response = client.responses.parse(
        model="gpt-5-nano",
        input=prompt,
        text_format=FaithfulnessResult,
    )

    if response.output_parsed is None:
        raise ValueError("Faithfulness Judge returned no parsed result")

    return response.output_parsed


def calculate_answer_faithfulness(
    context: str,
    answer: str,
) -> float:
    result = evaluate_faithfulness(
        context=context,
        answer=answer,
    )

    claim_verdicts = [
        claim.supported
        for claim in result.claims
    ]

    score = calculate_faithfulness_score(
        claim_verdicts
    )

    return score


def build_correctness_prompt(
    question: str,
    reference_answer: str,
    generated_answer: str,
) -> str:
    return f"""
You are an answer correctness evaluator for a RAG system.

Evaluate whether the generated answer correctly answers the question
based on the reference answer.

Rules:
- Evaluate whether the generated answer provides the
  essential information required by the question.
- Accept paraphrases and equivalent terminology.
- Mark the answer as correct if the core answer is correct.
- Do not penalize unrelated additional factual errors
  unless they contradict or undermine the core answer.
- Mark the answer as incorrect if essential information
  is missing, wrong, or contradicted.
- Explain your decision briefly.

Question:
{question}

Reference Answer:
{reference_answer}

Generated Answer:
{generated_answer}
"""



def evaluate_correctness(
    question: str,
    reference_answer: str,
    generated_answer: str,
) -> CorrectnessResult:
    prompt = build_correctness_prompt(
        question=question,
        reference_answer=reference_answer,
        generated_answer=generated_answer,
    )

    response = client.responses.parse(
        model="gpt-5-nano",
        input=prompt,
        text_format=CorrectnessResult,
    )

    if response.output_parsed is None:
        raise ValueError(
            "Correctness Judge returned no parsed result"
        )

    return response.output_parsed
