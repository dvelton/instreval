"""Eval runner — orchestrates prompt x instruction x run matrix."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from instreval import providers
from instreval.config import EvalConfig, DimensionSpec
from instreval.scorers.rule_based import RuleBasedResult, score_rule_based
from instreval.scorers.llm_judge import LLMJudgeResult, score_llm_judge


@dataclass
class SingleOutput:
    prompt_id: str
    instruction_set: str  # "baseline" or "candidate"
    run_number: int
    content: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class DimensionScore:
    dimension_name: str
    score: float
    detail: RuleBasedResult | LLMJudgeResult | None = None


@dataclass
class ScoredOutput:
    output: SingleOutput
    scores: list[DimensionScore] = field(default_factory=list)

    @property
    def avg_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)


@dataclass
class PromptResult:
    prompt_id: str
    prompt_text: str
    baseline_outputs: list[ScoredOutput]
    candidate_outputs: list[ScoredOutput]

    def baseline_avg(self, dimension: str | None = None) -> float:
        return self._avg(self.baseline_outputs, dimension)

    def candidate_avg(self, dimension: str | None = None) -> float:
        return self._avg(self.candidate_outputs, dimension)

    def _avg(self, outputs: list[ScoredOutput], dimension: str | None) -> float:
        if not outputs:
            return 0.0
        if dimension:
            scores = [
                s.score
                for o in outputs
                for s in o.scores
                if s.dimension_name == dimension
            ]
            return sum(scores) / len(scores) if scores else 0.0
        return sum(o.avg_score for o in outputs) / len(outputs)


@dataclass
class EvalResult:
    config: EvalConfig
    prompt_results: list[PromptResult]
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    elapsed_seconds: float = 0.0

    def baseline_overall(self, dimension: str | None = None) -> float:
        if not self.prompt_results:
            return 0.0
        return sum(
            pr.baseline_avg(dimension) for pr in self.prompt_results
        ) / len(self.prompt_results)

    def candidate_overall(self, dimension: str | None = None) -> float:
        if not self.prompt_results:
            return 0.0
        return sum(
            pr.candidate_avg(dimension) for pr in self.prompt_results
        ) / len(self.prompt_results)

    def to_dict(self) -> dict:
        """Serialize results for saving / passing to suggest."""
        results = []
        for pr in self.prompt_results:
            results.append({
                "prompt_id": pr.prompt_id,
                "prompt_text": pr.prompt_text,
                "baseline_avg": pr.baseline_avg(),
                "candidate_avg": pr.candidate_avg(),
                "baseline_outputs": [
                    {
                        "run": o.output.run_number,
                        "content": o.output.content,
                        "scores": [
                            {"dimension": s.dimension_name, "score": s.score}
                            for s in o.scores
                        ],
                    }
                    for o in pr.baseline_outputs
                ],
                "candidate_outputs": [
                    {
                        "run": o.output.run_number,
                        "content": o.output.content,
                        "scores": [
                            {"dimension": s.dimension_name, "score": s.score}
                            for s in o.scores
                        ],
                    }
                    for o in pr.candidate_outputs
                ],
            })
        return {
            "model": self.config.model,
            "baseline_name": self.config.instructions.baseline_name,
            "candidate_name": self.config.instructions.candidate_name,
            "baseline_overall": self.baseline_overall(),
            "candidate_overall": self.candidate_overall(),
            "prompt_results": results,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "elapsed_seconds": self.elapsed_seconds,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


async def _generate_output(
    model: str,
    system_prompt: str,
    user_prompt: str,
    prompt_id: str,
    instruction_set: str,
    run_number: int,
) -> SingleOutput:
    resp = await providers.complete(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    return SingleOutput(
        prompt_id=prompt_id,
        instruction_set=instruction_set,
        run_number=run_number,
        content=resp.content,
        prompt_tokens=resp.prompt_tokens,
        completion_tokens=resp.completion_tokens,
    )


async def _score_output(
    output: SingleOutput,
    dimensions: list[DimensionSpec],
    prompt_text: str,
    judge_model: str,
) -> ScoredOutput:
    scores: list[DimensionScore] = []

    for dim in dimensions:
        if dim.type == "rule_based":
            result = score_rule_based(dim, output.content)
            scores.append(DimensionScore(
                dimension_name=dim.name,
                score=result.score,
                detail=result,
            ))
        elif dim.type == "llm_judge":
            result = await score_llm_judge(
                dimension=dim,
                text=output.content,
                prompt_context=prompt_text,
                judge_model=judge_model,
            )
            scores.append(DimensionScore(
                dimension_name=dim.name,
                score=result.score,
                detail=result,
            ))

    return ScoredOutput(output=output, scores=scores)


async def run_eval(config: EvalConfig, on_progress=None) -> EvalResult:
    """Run a full evaluation: generate outputs and score them."""
    start = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    prompt_results: list[PromptResult] = []

    total_tasks = len(config.prompts) * config.runs * 2
    completed = 0

    for prompt_spec in config.prompts:
        baseline_scored: list[ScoredOutput] = []
        candidate_scored: list[ScoredOutput] = []

        # Generate all outputs for this prompt concurrently
        tasks = []
        for run_num in range(1, config.runs + 1):
            tasks.append(_generate_output(
                model=config.model,
                system_prompt=config.instructions.baseline,
                user_prompt=prompt_spec.text,
                prompt_id=prompt_spec.id,
                instruction_set="baseline",
                run_number=run_num,
            ))
            tasks.append(_generate_output(
                model=config.model,
                system_prompt=config.instructions.candidate,
                user_prompt=prompt_spec.text,
                prompt_id=prompt_spec.id,
                instruction_set="candidate",
                run_number=run_num,
            ))

        outputs = await asyncio.gather(*tasks)

        # Score all outputs
        score_tasks = []
        for output in outputs:
            total_prompt_tokens += output.prompt_tokens
            total_completion_tokens += output.completion_tokens
            score_tasks.append(_score_output(
                output=output,
                dimensions=config.dimensions,
                prompt_text=prompt_spec.text,
                judge_model=config.effective_judge_model,
            ))

        scored_outputs = await asyncio.gather(*score_tasks)

        for scored in scored_outputs:
            total_prompt_tokens += scored.output.prompt_tokens
            total_completion_tokens += scored.output.completion_tokens
            if scored.output.instruction_set == "baseline":
                baseline_scored.append(scored)
            else:
                candidate_scored.append(scored)

            completed += 1
            if on_progress:
                on_progress(completed, total_tasks)

        prompt_results.append(PromptResult(
            prompt_id=prompt_spec.id,
            prompt_text=prompt_spec.text,
            baseline_outputs=baseline_scored,
            candidate_outputs=candidate_scored,
        ))

    elapsed = time.time() - start

    return EvalResult(
        config=config,
        prompt_results=prompt_results,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        elapsed_seconds=elapsed,
    )
