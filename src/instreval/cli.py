"""CLI entry point — instreval compare, audit, suggest."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from instreval.config import (
    EvalConfig,
    InstructionsSpec,
    PromptSpec,
    DimensionSpec,
    load_eval_config,
    load_audit_config,
)
from instreval.runner import run_eval, EvalResult
from instreval.diagnosis import diagnose_results
from instreval.report import print_compare_report, print_audit_report

console = Console()


@click.group()
@click.version_option()
def cli():
    """instreval — A/B test and audit your AI custom instructions."""
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--save", "-s", type=click.Path(), help="Save results to JSON file")
def compare(config_path: str, save: str | None):
    """Compare two instruction sets using an eval config file.

    CONFIG_PATH is a YAML file defining instructions, prompts, and scoring.
    """
    try:
        config = load_eval_config(config_path)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        sys.exit(1)

    console.print(f"[bold]Running comparison eval...[/bold]")
    console.print(
        f"  Model: {config.model}  |  "
        f"Prompts: {len(config.prompts)}  |  "
        f"Runs: {config.runs}  |  "
        f"Dimensions: {len(config.dimensions)}"
    )
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating and scoring outputs...", total=None)

        def on_progress(completed, total):
            progress.update(task, description=f"Progress: {completed}/{total} outputs scored")

        result = asyncio.run(run_eval(config, on_progress=on_progress))

    diagnosis = diagnose_results(result)
    print_compare_report(result, diagnosis)

    if save:
        result.save(save)
        console.print(f"\n[dim]Results saved to {save}[/dim]")


@cli.command()
@click.argument("instructions_path", type=click.Path(exists=True))
@click.option("--model", "-m", required=True, help="Model to use (e.g., claude-sonnet-4-20250514)")
@click.option("--num-prompts", "-n", default=10, help="Number of test prompts to generate")
@click.option("--runs", "-r", default=3, help="Number of runs per prompt")
@click.option("--judge-model", "-j", help="Separate model for judging (defaults to --model)")
@click.option("--save", "-s", type=click.Path(), help="Save results to JSON file")
@click.option("--suggest/--no-suggest", default=True, help="Generate improvement suggestions")
def audit(
    instructions_path: str,
    model: str,
    num_prompts: int,
    runs: int,
    judge_model: str | None,
    save: str | None,
    suggest: bool,
):
    """Audit a single instruction set — auto-generates test prompts and scores.

    INSTRUCTIONS_PATH is a text file containing your custom instructions.
    """
    from instreval.prompt_gen import generate_test_prompts
    from instreval.suggest import generate_suggestions

    audit_config = load_audit_config(
        instructions_path, model, num_prompts, runs, judge_model
    )

    console.print(f"[bold]Auditing instructions: {audit_config.instructions_name}[/bold]")
    console.print(f"  Model: {model}  |  Generating {num_prompts} test prompts")
    console.print()

    # Step 1: Generate test prompts
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing instructions and generating test prompts...", total=None)
        gen_result = asyncio.run(
            generate_test_prompts(audit_config.instructions_text, model, num_prompts)
        )
        progress.update(task, description=f"Generated {len(gen_result['prompts'])} test prompts")

    prompts = [PromptSpec(id=p["id"], text=p["text"]) for p in gen_result["prompts"]]
    dimensions = [
        DimensionSpec(name=d["name"], type=d["type"], criteria=d.get("criteria"))
        for d in gen_result.get("dimensions", [])
    ]

    console.print(f"  Test prompts: {len(prompts)}")
    console.print(f"  Scoring dimensions: {', '.join(d.name for d in dimensions)}")
    console.print()

    # Build an EvalConfig for the runner (audit = baseline only, no candidate)
    eval_config = EvalConfig(
        model=model,
        instructions=InstructionsSpec(
            baseline=audit_config.instructions_text,
            candidate=audit_config.instructions_text,  # same — audit mode
            baseline_name=audit_config.instructions_name,
            candidate_name=audit_config.instructions_name,
        ),
        prompts=prompts,
        dimensions=dimensions,
        runs=runs,
        judge_model=judge_model,
    )

    # Step 2: Run eval
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running evaluation...", total=None)

        def on_progress(completed, total):
            progress.update(task, description=f"Progress: {completed}/{total} outputs scored")

        result = asyncio.run(run_eval(eval_config, on_progress=on_progress))

    diagnosis = diagnose_results(result)
    print_audit_report(result, diagnosis, audit_config.instructions_name)

    if save:
        result.save(save)
        console.print(f"\n[dim]Results saved to {save}[/dim]")

    # Step 3: Suggestions
    if suggest and diagnosis.has_issues:
        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating improvement suggestions...", total=None)
            suggestions = asyncio.run(generate_suggestions(
                diagnosis,
                audit_config.instructions_text,
                audit_config.effective_judge_model,
            ))

        console.print()
        console.print("[bold]Improvement Suggestions[/bold]")
        for i, s in enumerate(suggestions.suggestions, 1):
            conf_color = {"high": "green", "medium": "yellow", "low": "red"}.get(s.confidence, "white")
            console.print(f"\n  {i}. [{conf_color}][{s.confidence}][/{conf_color}] {s.dimension}")
            console.print(f"     Add: \"{s.instruction_to_add}\"", style="cyan")
            console.print(f"     Why: {s.rationale}", style="dim")

        if suggestions.suggested_instructions:
            suggested_path = Path(instructions_path).with_suffix(".suggested.txt")
            suggested_path.write_text(suggestions.suggested_instructions)
            console.print(f"\n[dim]Suggested instructions saved to {suggested_path}[/dim]")


@cli.command()
@click.argument("results_path", type=click.Path(exists=True))
@click.argument("instructions_path", type=click.Path(exists=True))
@click.option("--model", "-m", required=True, help="Model to use for generating suggestions")
def suggest(results_path: str, instructions_path: str, model: str):
    """Generate improvement suggestions from eval results.

    RESULTS_PATH is a JSON file from a previous compare or audit run (--save).
    INSTRUCTIONS_PATH is the instruction file to improve.
    """
    import json
    from instreval.suggest import generate_suggestions
    from instreval.diagnosis import Diagnosis, DiagnosisItem

    with open(results_path) as f:
        data = json.load(f)

    instructions_text = Path(instructions_path).read_text().strip()

    # Reconstruct a minimal diagnosis from saved results
    items = []
    weaknesses = []
    for pr in data.get("prompt_results", []):
        for output_group in [pr.get("baseline_outputs", []), pr.get("candidate_outputs", [])]:
            for o in output_group:
                for s in o.get("scores", []):
                    if s["score"] < 6.0:
                        items.append(DiagnosisItem(
                            dimension=s["dimension"],
                            instruction_set="target",
                            prompt_id=pr["prompt_id"],
                            score=s["score"],
                            explanation=f"Low score on {s['dimension']}",
                            evidence=[o.get("content", "")[:300]],
                        ))

    diagnosis = Diagnosis(items=items, weaknesses=weaknesses)

    if not diagnosis.has_issues:
        console.print("[green]No issues found — nothing to suggest.[/green]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating suggestions...", total=None)
        suggestion_set = asyncio.run(generate_suggestions(diagnosis, instructions_text, model))

    console.print()
    console.print("[bold]Improvement Suggestions[/bold]")
    for i, s in enumerate(suggestion_set.suggestions, 1):
        conf_color = {"high": "green", "medium": "yellow", "low": "red"}.get(s.confidence, "white")
        console.print(f"\n  {i}. [{conf_color}][{s.confidence}][/{conf_color}] {s.dimension}")
        console.print(f"     Add: \"{s.instruction_to_add}\"", style="cyan")
        console.print(f"     Why: {s.rationale}", style="dim")

    if suggestion_set.suggested_instructions:
        suggested_path = Path(instructions_path).with_suffix(".suggested.txt")
        suggested_path.write_text(suggestion_set.suggested_instructions)
        console.print(f"\n[dim]Suggested instructions saved to {suggested_path}[/dim]")


if __name__ == "__main__":
    cli()
