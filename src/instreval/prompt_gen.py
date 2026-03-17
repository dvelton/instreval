"""Auto-generate test prompts from instruction analysis (for audit mode)."""

from __future__ import annotations

import json
from instreval import providers


PROMPT_GEN_SYSTEM = """\
You are an expert at evaluating AI custom instructions. Given a set of custom \
instructions, your job is to generate test prompts that would effectively test \
whether an AI assistant is following those instructions.

For each instruction or rule in the custom instructions, generate a prompt that \
would naturally elicit behavior where that rule matters. The prompts should be \
realistic tasks a user would actually ask.

Also generate a set of scoring dimensions that capture what the instructions are \
trying to control.

Respond with ONLY valid JSON in this format:
{
  "prompts": [
    {"id": "short-kebab-id", "text": "The test prompt text", "targets": "What instruction this tests"}
  ],
  "dimensions": [
    {"name": "short-name", "type": "llm_judge", "criteria": "Scoring criteria text"}
  ]
}
"""


async def generate_test_prompts(
    instructions_text: str,
    model: str,
    num_prompts: int = 10,
) -> dict:
    """Analyze instructions and generate targeted test prompts and scoring dimensions."""
    user_prompt = (
        f"Generate {num_prompts} test prompts for these custom instructions. "
        f"Cover the most important rules and behaviors they define.\n\n"
        f"## Custom Instructions\n{instructions_text}"
    )

    response = await providers.complete(
        model=model,
        system_prompt=PROMPT_GEN_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.4,
        max_tokens=4096,
    )

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        import re
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response.content, re.DOTALL)
        if match:
            result = json.loads(match.group(1))
        else:
            raise ValueError(
                f"Could not parse prompt generation response as JSON. "
                f"Response: {response.content[:500]}"
            )

    if "prompts" not in result:
        raise ValueError("Generated response missing 'prompts' key")

    return result
