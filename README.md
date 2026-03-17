# instreval

A/B test and audit your AI custom instructions. Measure whether your instruction changes actually improve outputs, with data instead of vibes.

## What it does

instreval runs your prompts through an LLM with different instruction sets and scores the outputs on dimensions you define. It answers the question: "I changed my custom instructions -- did my outputs get better or worse?"

Three modes:

- **compare** -- A/B test two instruction sets against the same prompts and scoring criteria
- **audit** -- Evaluate a single instruction set; auto-generates test prompts and scoring dimensions
- **suggest** -- Generate specific instruction improvements based on eval results

## Install

```
pip install instreval
```

Or for development:

```
git clone https://github.com/dvelton/instreval.git
cd instreval
pip install -e ".[dev]"
```

## Quick start

### Audit mode (easiest way to start)

Point it at your custom instructions file. It generates test prompts, runs them, scores the output, and suggests improvements.

```
export ANTHROPIC_API_KEY=sk-...   # or OPENAI_API_KEY, etc.

instreval audit my-instructions.txt --model claude-sonnet-4-20250514
```

### Compare mode

Create an eval config (YAML) with two instruction sets, test prompts, and scoring criteria:

```yaml
model: "claude-sonnet-4-20250514"

instructions:
  baseline: baseline.txt
  candidate: candidate.txt

prompts:
  - id: memo-draft
    text: "Draft a memo summarizing the risks of adopting third-party AI models."
  - id: email-response
    text: "Write a response to a customer asking about our data retention policy."

scoring:
  dimensions:
    - name: conciseness
      type: llm_judge
      criteria: "Rate how concise the response is. Penalize filler and repetition."
    - name: ai_patterns
      type: rule_based
      rules:
        - pattern: " — "
          name: em_dash_overuse
          max_allowed: 2
        - pattern: "(?i)it's not .+—.?it's"
          name: negative_parallelism
          max_allowed: 0

runs: 3
```

Then run:

```
instreval compare eval.yaml --save results.json
```

### Suggest mode

Generate improvement suggestions from saved results:

```
instreval suggest results.json my-instructions.txt --model claude-sonnet-4-20250514
```

## Scoring types

**rule_based** -- Pattern matching with regex. Runs locally, zero API cost. Good for catching specific text patterns (em dash overuse, rhetorical structures, formatting violations).

**llm_judge** -- Uses an LLM to score outputs on subjective criteria you define. Costs API calls. Good for harder-to-quantify dimensions like conciseness, tone, actionability, naturalness.

## Model support

instreval uses [LiteLLM](https://github.com/BerriAI/litellm) under the hood, so it works with any provider:

- OpenAI: `gpt-4o`, `gpt-4o-mini`, etc.
- Anthropic: `claude-sonnet-4-20250514`, `claude-haiku-4-20250514`, etc.
- Google: `gemini/gemini-pro`, etc.
- Azure, AWS Bedrock, Ollama, and [100+ more](https://docs.litellm.ai/docs/providers)

Set the appropriate API key as an environment variable and specify the model string.

## How it works

1. Loads your instruction sets and eval config
2. Runs each test prompt against each instruction set N times (configurable, default 3)
3. Scores each output on every dimension (rule-based checks run locally; LLM-judge calls the model)
4. Produces a comparison report with per-dimension and per-prompt breakdowns
5. Diagnoses weaknesses with specific evidence
6. Optionally generates improvement suggestions with rationale

Multiple runs per prompt account for output variance -- AI responses aren't deterministic, so a single sample isn't reliable.

## Example

See `examples/writing-style/` for a complete example that tests whether adding anti-AI-slop writing rules to instructions actually reduces AI writing patterns in outputs.

## Contributing

PRs welcome. Areas where contributions would be particularly useful:

- New scorer types (readability metrics, sentiment analysis, factual consistency)
- Preset scoring dimensions for common use cases
- Better report formats (markdown export, HTML)
- Preset test prompt libraries for common instruction categories

## License

MIT
