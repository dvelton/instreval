"""Load and validate eval configuration files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RuleSpec:
    pattern: str
    name: str
    max_allowed: int = 0
    _compiled: re.Pattern | None = field(default=None, repr=False)

    @property
    def compiled(self) -> re.Pattern:
        if self._compiled is None:
            self._compiled = re.compile(self.pattern)
        return self._compiled


@dataclass
class DimensionSpec:
    name: str
    type: str  # "llm_judge" or "rule_based"
    criteria: str | None = None  # for llm_judge
    rules: list[RuleSpec] = field(default_factory=list)  # for rule_based


@dataclass
class PromptSpec:
    id: str
    text: str


@dataclass
class InstructionsSpec:
    baseline: str  # text content
    candidate: str  # text content
    baseline_name: str = "baseline"
    candidate_name: str = "candidate"


@dataclass
class EvalConfig:
    model: str
    instructions: InstructionsSpec
    prompts: list[PromptSpec]
    dimensions: list[DimensionSpec]
    runs: int = 3
    judge_model: str | None = None  # separate model for judging; defaults to model

    @property
    def effective_judge_model(self) -> str:
        return self.judge_model or self.model


@dataclass
class AuditConfig:
    model: str
    instructions_text: str
    instructions_name: str = "instructions"
    num_prompts: int = 10
    runs: int = 3
    judge_model: str | None = None

    @property
    def effective_judge_model(self) -> str:
        return self.judge_model or self.model


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = base_dir / p
    return p


def _parse_rules(raw_rules: list[dict[str, Any]]) -> list[RuleSpec]:
    rules = []
    for r in raw_rules:
        rules.append(RuleSpec(
            pattern=r["pattern"],
            name=r.get("name", "unnamed_rule"),
            max_allowed=r.get("max_allowed", 0),
        ))
    return rules


def _parse_dimensions(raw: list[dict[str, Any]]) -> list[DimensionSpec]:
    dims = []
    for d in raw:
        dim_type = d["type"]
        dim = DimensionSpec(
            name=d["name"],
            type=dim_type,
            criteria=d.get("criteria"),
            rules=_parse_rules(d.get("rules", [])) if dim_type == "rule_based" else [],
        )
        dims.append(dim)
    return dims


def load_eval_config(config_path: str | Path) -> EvalConfig:
    """Load an eval config from a YAML file."""
    config_path = Path(config_path)
    base_dir = config_path.parent

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError(f"Empty config file: {config_path}")

    instr_raw = raw["instructions"]
    baseline_path = _resolve_path(instr_raw["baseline"], base_dir)
    candidate_path = _resolve_path(instr_raw["candidate"], base_dir)

    baseline_text = baseline_path.read_text().strip()
    candidate_text = candidate_path.read_text().strip()

    prompts = [
        PromptSpec(id=p["id"], text=p["text"])
        for p in raw["prompts"]
    ]

    scoring_raw = raw.get("scoring", {})
    dimensions = _parse_dimensions(scoring_raw.get("dimensions", []))

    return EvalConfig(
        model=raw["model"],
        instructions=InstructionsSpec(
            baseline=baseline_text,
            candidate=candidate_text,
            baseline_name=instr_raw.get("baseline_name", baseline_path.stem),
            candidate_name=instr_raw.get("candidate_name", candidate_path.stem),
        ),
        prompts=prompts,
        dimensions=dimensions,
        runs=raw.get("runs", 3),
        judge_model=raw.get("judge_model"),
    )


def load_audit_config(
    instructions_path: str | Path,
    model: str,
    num_prompts: int = 10,
    runs: int = 3,
    judge_model: str | None = None,
) -> AuditConfig:
    """Load a single instructions file for audit mode."""
    instructions_path = Path(instructions_path)
    text = instructions_path.read_text().strip()

    if not text:
        raise ValueError(f"Empty instructions file: {instructions_path}")

    return AuditConfig(
        model=model,
        instructions_text=text,
        instructions_name=instructions_path.stem,
        num_prompts=num_prompts,
        runs=runs,
        judge_model=judge_model,
    )
