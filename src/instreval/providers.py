"""LiteLLM provider wrapper for model calls."""

from __future__ import annotations

from dataclasses import dataclass

import litellm

# Suppress LiteLLM's verbose logging
litellm.suppress_debug_info = True


@dataclass
class ModelResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int


async def complete(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> ModelResponse:
    """Call an LLM with a system prompt (instructions) and user prompt."""
    response = await litellm.acompletion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    choice = response.choices[0]
    usage = response.usage

    return ModelResponse(
        content=choice.message.content or "",
        model=response.model or model,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )


async def judge(
    model: str,
    criteria: str,
    text_to_judge: str,
    prompt_context: str,
) -> tuple[float, str]:
    """Use an LLM to score text on a given criteria. Returns (score, reasoning)."""
    system = (
        "You are an expert evaluator. Score the following AI-generated response "
        "on the given criteria. Respond with ONLY valid JSON in this exact format:\n"
        '{"score": <number 1-10>, "reasoning": "<brief explanation>"}\n\n'
        "Score scale: 1 = very poor, 5 = adequate, 10 = excellent."
    )

    user = (
        f"## Criteria\n{criteria}\n\n"
        f"## Original Prompt\n{prompt_context}\n\n"
        f"## Response to Evaluate\n{text_to_judge}"
    )

    response = await complete(
        model=model,
        system_prompt=system,
        user_prompt=user,
        temperature=0.1,
        max_tokens=512,
    )

    import json
    try:
        result = json.loads(response.content)
        score = float(result["score"])
        reasoning = result.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, ValueError):
        # Try to extract score from non-JSON response
        import re
        match = re.search(r'"?score"?\s*[:=]\s*(\d+(?:\.\d+)?)', response.content)
        if match:
            score = float(match.group(1))
            reasoning = response.content
        else:
            score = 5.0
            reasoning = f"Could not parse judge response: {response.content[:200]}"

    score = max(1.0, min(10.0, score))
    return score, reasoning
