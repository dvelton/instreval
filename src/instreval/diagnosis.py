"""Diagnosis engine — analyze low scores and explain why."""

from __future__ import annotations

from dataclasses import dataclass, field

from instreval.runner import EvalResult, ScoredOutput, PromptResult
from instreval.scorers.rule_based import RuleBasedResult
from instreval.scorers.llm_judge import LLMJudgeResult


@dataclass
class DiagnosisItem:
    dimension: str
    instruction_set: str
    prompt_id: str
    score: float
    explanation: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class Diagnosis:
    items: list[DiagnosisItem] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return len(self.items) > 0


LOW_SCORE_THRESHOLD = 6.0
HIGH_SCORE_THRESHOLD = 8.0


def _diagnose_rule_based(
    result: RuleBasedResult,
    output: ScoredOutput,
) -> DiagnosisItem | None:
    if result.score >= LOW_SCORE_THRESHOLD:
        return None

    evidence = []
    explanations = []
    for v in result.violations:
        excess = v.count - v.max_allowed
        explanations.append(
            f"'{v.rule_name}' triggered {v.count} times "
            f"(max allowed: {v.max_allowed}, {excess} over limit)"
        )
        evidence.extend(v.matches[:3])

    return DiagnosisItem(
        dimension=result.dimension_name,
        instruction_set=output.output.instruction_set,
        prompt_id=output.output.prompt_id,
        score=result.score,
        explanation="; ".join(explanations),
        evidence=evidence,
    )


def _diagnose_llm_judge(
    result: LLMJudgeResult,
    output: ScoredOutput,
) -> DiagnosisItem | None:
    if result.score >= LOW_SCORE_THRESHOLD:
        return None

    # Pull a representative excerpt from the output
    content = output.output.content
    excerpt = content[:300] + "..." if len(content) > 300 else content

    return DiagnosisItem(
        dimension=result.dimension_name,
        instruction_set=output.output.instruction_set,
        prompt_id=output.output.prompt_id,
        score=result.score,
        explanation=result.reasoning,
        evidence=[excerpt],
    )


def diagnose_results(result: EvalResult) -> Diagnosis:
    """Analyze eval results to identify strengths and weaknesses."""
    items: list[DiagnosisItem] = []
    dim_scores: dict[str, dict[str, list[float]]] = {}

    for pr in result.prompt_results:
        all_outputs = pr.baseline_outputs + pr.candidate_outputs
        for scored in all_outputs:
            for ds in scored.scores:
                key = (ds.dimension_name, scored.output.instruction_set)
                if key not in dim_scores:
                    dim_scores[key] = {"scores": []}
                dim_scores[(ds.dimension_name, scored.output.instruction_set)]["scores"].append(ds.score)

                # Diagnose individual low scores
                if ds.detail is None:
                    continue

                if isinstance(ds.detail, RuleBasedResult):
                    item = _diagnose_rule_based(ds.detail, scored)
                elif isinstance(ds.detail, LLMJudgeResult):
                    item = _diagnose_llm_judge(ds.detail, scored)
                else:
                    item = None

                if item:
                    items.append(item)

    # Identify overall strengths and weaknesses by dimension x instruction_set
    strengths = []
    weaknesses = []
    for (dim, instr_set), data in dim_scores.items():
        scores = data["scores"]
        avg = sum(scores) / len(scores) if scores else 0
        label = f"{dim} ({instr_set})"
        if avg >= HIGH_SCORE_THRESHOLD:
            strengths.append(f"{label}: avg {avg:.1f}/10")
        elif avg < LOW_SCORE_THRESHOLD:
            weaknesses.append(f"{label}: avg {avg:.1f}/10")

    return Diagnosis(items=items, strengths=strengths, weaknesses=weaknesses)
