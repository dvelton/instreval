"""LLM-as-judge scorer — uses an LLM to evaluate outputs on given criteria."""

from __future__ import annotations

from dataclasses import dataclass

from instreval.config import DimensionSpec
from instreval import providers


@dataclass
class LLMJudgeResult:
    dimension_name: str
    score: float  # 1-10 scale
    reasoning: str


async def score_llm_judge(
    dimension: DimensionSpec,
    text: str,
    prompt_context: str,
    judge_model: str,
) -> LLMJudgeResult:
    """Score text using an LLM judge on the dimension's criteria."""
    if not dimension.criteria:
        raise ValueError(
            f"Dimension '{dimension.name}' is type llm_judge but has no criteria"
        )

    score, reasoning = await providers.judge(
        model=judge_model,
        criteria=dimension.criteria,
        text_to_judge=text,
        prompt_context=prompt_context,
    )

    return LLMJudgeResult(
        dimension_name=dimension.name,
        score=score,
        reasoning=reasoning,
    )
