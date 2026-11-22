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


class ClaimExtractionResult(BaseModel):
    claims: list[str]


class ClaimVerificationResult(BaseModel):
    verdicts: list[ClaimVerdict]


def calculate_faithfulness_score(claim_verdicts: list[bool]) -> float:
    if not claim_verdicts:
        return 0.0
    return sum(claim_verdicts) / len(claim_verdicts)


def build_faithfulness_prompt(context: str, answer: str) -> str:
    return f"""
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


def evaluate_faithfulness(context: str, answer: str) -> FaithfulnessResult:
    prompt = build_faithfulness_prompt(context=context, answer=answer)
    response = client.responses.parse(
        model="gpt-5-nano", input=prompt, text_format=FaithfulnessResult
    )
    if response.output_parsed is None:
        raise ValueError("Faithfulness Judge returned no parsed result")
    return response.output_parsed


def calculate_answer_faithfulness(context: str, answer: str) -> float:
    result = evaluate_faithfulness(context=context, answer=answer)
    return calculate_faithfulness_score(
        [claim.supported for claim in result.claims]
    )


def build_claim_extraction_prompt(answer: str) -> str:
    return f"""
You are a factual claim extractor for a RAG evaluation system.

Extract every factual claim expressed in the generated answer.

Rules:
- Use ONLY the generated answer; do not add external facts.
- Split compound statements into minimal, independently verifiable claims.
- Preserve all factual information; do not omit acronym expansions,
  identities, relationships, dates, or architectural properties.
- Treat acronym expansions as separate claims. For example,
  'Dense Passage Retriever (DPR)' entails the separate claim
  'DPR stands for Dense Passage Retriever.'
- Resolve pronouns where needed so each claim is understandable alone.
- Do not judge whether claims are true or supported.
- Avoid duplicates and return an empty list if there are no factual claims.

Generated Answer:
{answer}
"""


def extract_claims(answer: str) -> list[str]:
    response = client.responses.parse(
        model="gpt-5-nano",
        input=build_claim_extraction_prompt(answer),
        text_format=ClaimExtractionResult,
    )
    if response.output_parsed is None:
        raise ValueError("Claim Extractor returned no parsed result")
    return response.output_parsed.claims


def build_claim_verification_prompt(context: str, claims: list[str]) -> str:
    numbered_claims = "\n".join(
        f"{index}. {claim}" for index, claim in enumerate(claims, start=1)
    )
    return f"""
You are a strict claim verifier for a RAG evaluation system.

For EACH numbered claim below, determine whether the context supports it.

Rules:
- Use ONLY the provided context, never outside knowledge.
- Support requires explicit evidence or a logical entailment.
- Mentioning an acronym does NOT establish its expanded form.
- Unsupported means insufficient evidence, not necessarily false.
- Do not merge, split, rewrite, add, or omit claims.
- Return exactly one verdict per input claim, in the same order.
- Copy each claim text exactly into its verdict.
- Explain each decision briefly. Quote relevant context evidence when supported;
  identify missing evidence when unsupported.

Context:
{context}

Claims:
{numbered_claims}
"""


def verify_claims(context: str, claims: list[str]) -> FaithfulnessResult:
    if not claims:
        return FaithfulnessResult(claims=[])

    response = client.responses.parse(
        model="gpt-5-nano",
        input=build_claim_verification_prompt(context, claims),
        text_format=ClaimVerificationResult,
    )
    if response.output_parsed is None:
        raise ValueError("Claim Verifier returned no parsed result")

    verdicts = response.output_parsed.verdicts
    if len(verdicts) != len(claims) or any(
        verdict.claim != claim for verdict, claim in zip(verdicts, claims)
    ):
        raise ValueError("Claim Verifier changed the claim list or order")
    return FaithfulnessResult(claims=verdicts)


def evaluate_faithfulness_separately(
    context: str, answer: str
) -> FaithfulnessResult:
    claims = extract_claims(answer)
    return verify_claims(context=context, claims=claims)


def build_correctness_prompt(
    question: str, reference_answer: str, generated_answer: str
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
    question: str, reference_answer: str, generated_answer: str
) -> CorrectnessResult:
    prompt = build_correctness_prompt(
        question=question,
        reference_answer=reference_answer,
        generated_answer=generated_answer,
    )
    response = client.responses.parse(
        model="gpt-5-nano", input=prompt, text_format=CorrectnessResult
    )
    if response.output_parsed is None:
        raise ValueError("Correctness Judge returned no parsed result")
    return response.output_parsed
