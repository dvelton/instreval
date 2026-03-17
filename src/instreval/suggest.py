"""Generate instruction improvement suggestions from diagnosis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from instreval import providers
from instreval.diagnosis import Diagnosis


@dataclass
class Suggestion:
    dimension: str
    instruction_to_add: str
    rationale: str
    confidence: str  # "high", "medium", "low"


@dataclass
class SuggestionSet:
    suggestions: list[Suggestion] = field(default_factory=list)
    original_instructions: str = ""
    suggested_instructions: str = ""


SUGGEST_SYSTEM = """\
You are an expert at writing custom instructions for AI assistants. Given a \
diagnosis of weaknesses in an AI's outputs when using certain custom instructions, \
generate specific instruction additions or modifications that would address the \
identified issues.

For each suggestion:
- Be specific and actionable (not "be more concise" but "keep responses under 400 \
words unless the task requires detailed analysis")
- Explain why this change would help
- Rate your confidence (high/medium/low)

Respond with ONLY valid JSON:
{
  "suggestions": [
    {
      "dimension": "dimension name this addresses",
      "instruction_to_add": "The exact text to add to the instructions",
      "rationale": "Why this would improve scores on this dimension",
      "confidence": "high|medium|low"
    }
  ],
  "suggested_instructions": "The full revised instructions text incorporating all suggestions"
}
"""


async def generate_suggestions(
    diagnosis: Diagnosis,
    original_instructions: str,
    model: str,
) -> SuggestionSet:
    """Generate instruction improvement suggestions from a diagnosis."""
    weakness_text = "\n".join(f"- {w}" for w in diagnosis.weaknesses)
    issue_text = "\n".join(
        f"- [{item.dimension}] score {item.score:.1f}: {item.explanation}"
        for item in diagnosis.items[:15]  # cap to avoid token limits
    )
    evidence_text = "\n".join(
        f'  "{item.evidence[0][:200]}"'
        for item in diagnosis.items[:5]
        if item.evidence
    )

    user_prompt = (
        f"## Current Instructions\n{original_instructions}\n\n"
        f"## Weaknesses Identified\n{weakness_text}\n\n"
        f"## Specific Issues\n{issue_text}\n\n"
        f"## Example Evidence\n{evidence_text}\n\n"
        f"Generate specific instruction improvements to address these weaknesses."
    )

    response = await providers.complete(
        model=model,
        system_prompt=SUGGEST_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.3,
        max_tokens=4096,
    )

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        import re
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response.content, re.DOTALL)
        if match:
            result = json.loads(match.group(1))
        else:
            raise ValueError(f"Could not parse suggestions: {response.content[:500]}")

    suggestions = [
        Suggestion(
            dimension=s.get("dimension", "general"),
            instruction_to_add=s["instruction_to_add"],
            rationale=s.get("rationale", ""),
            confidence=s.get("confidence", "medium"),
        )
        for s in result.get("suggestions", [])
    ]

    return SuggestionSet(
        suggestions=suggestions,
        original_instructions=original_instructions,
        suggested_instructions=result.get("suggested_instructions", ""),
    )
