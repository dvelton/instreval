"""Rule-based scorers — pattern matching, length checks, counts."""

from __future__ import annotations

from dataclasses import dataclass

from instreval.config import DimensionSpec, RuleSpec


@dataclass
class RuleViolation:
    rule_name: str
    pattern: str
    count: int
    max_allowed: int
    matches: list[str]  # the actual matched text snippets


@dataclass
class RuleBasedResult:
    dimension_name: str
    score: float  # 1-10 scale
    violations: list[RuleViolation]
    total_violations: int


def _score_from_violations(violations: list[RuleViolation]) -> float:
    """Convert violation counts into a 1-10 score. Fewer violations = higher score."""
    if not violations:
        return 10.0

    total_excess = sum(
        max(0, v.count - v.max_allowed) for v in violations
    )

    if total_excess == 0:
        return 10.0

    # Each excess violation drops the score. Diminishing penalty past 5 violations.
    penalty = min(total_excess * 1.5, 9.0)
    return max(1.0, 10.0 - penalty)


def check_rule(rule: RuleSpec, text: str) -> RuleViolation:
    """Check a single rule against text."""
    matches_iter = rule.compiled.finditer(text)
    matched_texts = []
    for m in matches_iter:
        # Grab surrounding context (up to 60 chars on each side)
        start = max(0, m.start() - 60)
        end = min(len(text), m.end() + 60)
        snippet = text[start:end].replace("\n", " ").strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        matched_texts.append(snippet)

    return RuleViolation(
        rule_name=rule.name,
        pattern=rule.pattern,
        count=len(matched_texts),
        max_allowed=rule.max_allowed,
        matches=matched_texts,
    )


def score_rule_based(dimension: DimensionSpec, text: str) -> RuleBasedResult:
    """Score text against all rules in a rule_based dimension."""
    violations = []
    for rule in dimension.rules:
        violation = check_rule(rule, text)
        if violation.count > violation.max_allowed:
            violations.append(violation)

    return RuleBasedResult(
        dimension_name=dimension.name,
        score=_score_from_violations(violations),
        violations=violations,
        total_violations=sum(v.count for v in violations),
    )
