"""Tests for config loading."""

import tempfile
from pathlib import Path

import pytest

from instreval.config import load_eval_config, load_audit_config


@pytest.fixture
def sample_eval_dir(tmp_path):
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("You are a helpful assistant.")

    candidate = tmp_path / "candidate.txt"
    candidate.write_text("You are a helpful assistant. Be concise.")

    config = tmp_path / "eval.yaml"
    config.write_text("""
model: "gpt-4o-mini"

instructions:
  baseline: baseline.txt
  candidate: candidate.txt

prompts:
  - id: test-prompt
    text: "Write a short summary."

scoring:
  dimensions:
    - name: conciseness
      type: llm_judge
      criteria: "Rate conciseness."
    - name: patterns
      type: rule_based
      rules:
        - pattern: " — "
          name: em_dash
          max_allowed: 2

runs: 2
""")
    return tmp_path


def test_load_eval_config(sample_eval_dir):
    config = load_eval_config(sample_eval_dir / "eval.yaml")

    assert config.model == "gpt-4o-mini"
    assert config.instructions.baseline == "You are a helpful assistant."
    assert config.instructions.candidate == "You are a helpful assistant. Be concise."
    assert len(config.prompts) == 1
    assert config.prompts[0].id == "test-prompt"
    assert len(config.dimensions) == 2
    assert config.runs == 2


def test_load_eval_config_dimensions(sample_eval_dir):
    config = load_eval_config(sample_eval_dir / "eval.yaml")

    llm_dim = next(d for d in config.dimensions if d.type == "llm_judge")
    assert llm_dim.name == "conciseness"
    assert llm_dim.criteria == "Rate conciseness."

    rule_dim = next(d for d in config.dimensions if d.type == "rule_based")
    assert rule_dim.name == "patterns"
    assert len(rule_dim.rules) == 1
    assert rule_dim.rules[0].name == "em_dash"
    assert rule_dim.rules[0].max_allowed == 2


def test_load_audit_config(tmp_path):
    instructions = tmp_path / "instructions.txt"
    instructions.write_text("Be concise and direct.")

    config = load_audit_config(instructions, model="gpt-4o-mini")

    assert config.model == "gpt-4o-mini"
    assert config.instructions_text == "Be concise and direct."
    assert config.instructions_name == "instructions"
    assert config.num_prompts == 10


def test_empty_instructions_raises(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")

    with pytest.raises(ValueError, match="Empty instructions"):
        load_audit_config(empty, model="gpt-4o-mini")
