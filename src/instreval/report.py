"""Terminal reports using Rich — compare and audit modes."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from instreval.runner import EvalResult
from instreval.diagnosis import Diagnosis


console = Console()


def _score_color(score: float) -> str:
    if score >= 8.0:
        return "green"
    elif score >= 6.0:
        return "yellow"
    return "red"


def _delta_str(baseline: float, candidate: float) -> Text:
    delta = candidate - baseline
    if abs(delta) < 0.1:
        return Text(f"  {delta:+.1f}", style="dim")
    elif delta > 0:
        return Text(f" +{delta:.1f}", style="bold green")
    else:
        return Text(f" {delta:.1f}", style="bold red")


def print_compare_report(result: EvalResult, diagnosis: Diagnosis) -> None:
    """Print a full comparison report to the terminal."""
    b_name = result.config.instructions.baseline_name
    c_name = result.config.instructions.candidate_name

    console.print()
    console.print(
        Panel(
            f"[bold]instreval[/bold] compare report\n"
            f"Model: {result.config.model}  |  "
            f"Prompts: {len(result.prompt_results)}  |  "
            f"Runs: {result.config.runs}  |  "
            f"Time: {result.elapsed_seconds:.1f}s",
            border_style="blue",
        )
    )

    # Overall scores table
    table = Table(title="Overall Scores", show_header=True, header_style="bold")
    table.add_column("Dimension", style="cyan")
    table.add_column(b_name, justify="right")
    table.add_column(c_name, justify="right")
    table.add_column("Delta", justify="right")

    dims = set()
    for pr in result.prompt_results:
        for so in pr.baseline_outputs + pr.candidate_outputs:
            for s in so.scores:
                dims.add(s.dimension_name)

    for dim in sorted(dims):
        b_score = result.baseline_overall(dim)
        c_score = result.candidate_overall(dim)
        table.add_row(
            dim,
            Text(f"{b_score:.1f}", style=_score_color(b_score)),
            Text(f"{c_score:.1f}", style=_score_color(c_score)),
            _delta_str(b_score, c_score),
        )

    # Overall row
    b_overall = result.baseline_overall()
    c_overall = result.candidate_overall()
    table.add_row(
        Text("OVERALL", style="bold"),
        Text(f"{b_overall:.1f}", style=f"bold {_score_color(b_overall)}"),
        Text(f"{c_overall:.1f}", style=f"bold {_score_color(c_overall)}"),
        _delta_str(b_overall, c_overall),
    )

    console.print(table)

    # Per-prompt breakdown
    console.print()
    prompt_table = Table(title="Per-Prompt Scores", show_header=True, header_style="bold")
    prompt_table.add_column("Prompt", style="cyan")
    prompt_table.add_column(b_name, justify="right")
    prompt_table.add_column(c_name, justify="right")
    prompt_table.add_column("Delta", justify="right")

    for pr in result.prompt_results:
        b = pr.baseline_avg()
        c = pr.candidate_avg()
        prompt_table.add_row(
            pr.prompt_id,
            Text(f"{b:.1f}", style=_score_color(b)),
            Text(f"{c:.1f}", style=_score_color(c)),
            _delta_str(b, c),
        )

    console.print(prompt_table)

    # Diagnosis
    if diagnosis.strengths:
        console.print()
        console.print("[bold green]Strengths[/bold green]")
        for s in diagnosis.strengths:
            console.print(f"  + {s}")

    if diagnosis.weaknesses:
        console.print()
        console.print("[bold red]Weaknesses[/bold red]")
        for w in diagnosis.weaknesses:
            console.print(f"  - {w}")

    if diagnosis.items:
        console.print()
        console.print(f"[bold yellow]Diagnosis[/bold yellow] ({len(diagnosis.items)} issues found)")
        for item in diagnosis.items[:10]:  # limit to top 10
            console.print(
                f"  [{item.instruction_set}] {item.dimension} "
                f"(prompt: {item.prompt_id}, score: {item.score:.1f})"
            )
            console.print(f"    {item.explanation}", style="dim")
            if item.evidence:
                console.print(f'    Evidence: "{item.evidence[0][:120]}..."', style="dim italic")

    # Token usage
    console.print()
    console.print(
        f"[dim]Tokens used: {result.total_prompt_tokens:,} prompt + "
        f"{result.total_completion_tokens:,} completion[/dim]"
    )


def print_audit_report(
    result: EvalResult,
    diagnosis: Diagnosis,
    instructions_name: str,
) -> None:
    """Print an audit report for a single instruction set."""
    console.print()
    console.print(
        Panel(
            f"[bold]instreval[/bold] audit report\n"
            f"Instructions: {instructions_name}  |  "
            f"Model: {result.config.model}  |  "
            f"Prompts: {len(result.prompt_results)}  |  "
            f"Time: {result.elapsed_seconds:.1f}s",
            border_style="blue",
        )
    )

    # Dimension scores
    table = Table(title="Dimension Scores", show_header=True, header_style="bold")
    table.add_column("Dimension", style="cyan")
    table.add_column("Score", justify="right")

    dims = set()
    for pr in result.prompt_results:
        for so in pr.baseline_outputs:
            for s in so.scores:
                dims.add(s.dimension_name)

    for dim in sorted(dims):
        score = result.baseline_overall(dim)
        table.add_row(dim, Text(f"{score:.1f}", style=_score_color(score)))

    overall = result.baseline_overall()
    table.add_row(
        Text("OVERALL", style="bold"),
        Text(f"{overall:.1f}", style=f"bold {_score_color(overall)}"),
    )

    console.print(table)

    if diagnosis.strengths:
        console.print()
        console.print("[bold green]Strengths[/bold green]")
        for s in diagnosis.strengths:
            console.print(f"  + {s}")

    if diagnosis.weaknesses:
        console.print()
        console.print("[bold red]Weaknesses[/bold red]")
        for w in diagnosis.weaknesses:
            console.print(f"  - {w}")

    if diagnosis.items:
        console.print()
        console.print(f"[bold yellow]Issues[/bold yellow] ({len(diagnosis.items)} found)")
        for item in diagnosis.items[:10]:
            console.print(
                f"  {item.dimension} (prompt: {item.prompt_id}, score: {item.score:.1f})"
            )
            console.print(f"    {item.explanation}", style="dim")

    console.print()
    console.print(
        f"[dim]Tokens: {result.total_prompt_tokens:,} prompt + "
        f"{result.total_completion_tokens:,} completion[/dim]"
    )
