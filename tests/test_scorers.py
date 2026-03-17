"""Tests for rule-based scorers."""

import pytest

from instreval.config import DimensionSpec, RuleSpec
from instreval.scorers.rule_based import score_rule_based, check_rule


def _make_rule_dimension(rules):
    return DimensionSpec(
        name="test_dim",
        type="rule_based",
        rules=[RuleSpec(**r) for r in rules],
    )


def test_no_violations():
    dim = _make_rule_dimension([
        {"pattern": " — ", "name": "em_dash", "max_allowed": 2},
    ])
    result = score_rule_based(dim, "This is clean text with no em dashes.")
    assert result.score == 10.0
    assert result.violations == []


def test_within_allowed():
    dim = _make_rule_dimension([
        {"pattern": " — ", "name": "em_dash", "max_allowed": 2},
    ])
    result = score_rule_based(dim, "One — dash and two — dashes are fine.")
    assert result.score == 10.0
    assert result.violations == []


def test_exceeds_allowed():
    dim = _make_rule_dimension([
        {"pattern": " — ", "name": "em_dash", "max_allowed": 2},
    ])
    text = "First — second — third — fourth — fifth — overkill."
    result = score_rule_based(dim, text)
    assert result.score < 10.0
    assert len(result.violations) == 1
    assert result.violations[0].rule_name == "em_dash"
    assert result.violations[0].count == 5


def test_zero_allowed():
    dim = _make_rule_dimension([
        {"pattern": "(?i)it's not .+—.?it's", "name": "neg_parallelism", "max_allowed": 0},
    ])
    text = "It's not about speed—it's about quality."
    result = score_rule_based(dim, text)
    assert result.score < 10.0
    assert result.violations[0].count == 1


def test_multiple_rules():
    dim = _make_rule_dimension([
        {"pattern": " — ", "name": "em_dash", "max_allowed": 0},
        {"pattern": "(?i)let's break this down", "name": "pedagogical", "max_allowed": 0},
    ])
    text = "Let's break this down — it's complex — very complex."
    result = score_rule_based(dim, text)
    assert len(result.violations) == 2


def test_check_rule_captures_context():
    rule = RuleSpec(pattern=" — ", name="em_dash", max_allowed=0)
    violation = check_rule(rule, "The problem — and this matters — is systemic.")
    assert violation.count == 2
    assert len(violation.matches) == 2
    # Each match should contain surrounding context
    assert any("problem" in m for m in violation.matches)


def test_clean_text_perfect_score():
    dim = _make_rule_dimension([
        {"pattern": " — ", "name": "em_dash", "max_allowed": 2},
        {"pattern": "(?i)let's break this down", "name": "pedagogical", "max_allowed": 0},
        {"pattern": "(?i)it's worth noting", "name": "filler", "max_allowed": 0},
    ])
    text = (
        "The proposal has three components. First, we reduce API latency by "
        "caching responses at the edge. Second, we consolidate the auth layer "
        "into a single service. Third, we migrate the database to a managed "
        "provider. Expected timeline is six weeks."
    )
    result = score_rule_based(dim, text)
    assert result.score == 10.0
