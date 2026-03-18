from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from irbg.config import ConfigError, load_models_config
from irbg.db.operations import DbConfig, connect, list_benchmark_runs
from irbg.db.schema import create_tables
from irbg.demographics import DemographicsError, get_variant_group
from irbg.engine.prompt_builder import PromptBuildError
from irbg.engine.provider import OpenRouterClient
from irbg.engine.runner import (
    run_all_template_variants,
    run_single_scenario,
    run_single_template_variant,
    run_template_folder,
)
from irbg.engine.variant_generator import (
    VariantGenerationError,
    generate_single_prompt_for_variant,
)
from irbg.scenarios.discovery import ScenarioDiscoveryError
from irbg.scenarios.template_loader import (
    ScenarioTemplateLoadError,
    load_scenario_template,
)
from irbg.scoring.p1 import P1ScoringError, score_p1_run

console = Console()


@click.group()
def main() -> None:
    """
    IRBG (Institutional Readiness & Bias Benchmark for Governance)

    CLI for database setup, provider checks, benchmark execution,
    and scoring.
    """
    load_dotenv()


def _ensure_database(db_path: Path) -> None:
    db = DbConfig(path=db_path)
    conn = connect(db)

    try:
        create_tables(conn)
    finally:
        conn.close()


@main.command("init-db")
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    default=Path("./irbg.sqlite"),
    show_default=True,
    help="SQLite database file path.",
)
def init_db(db_path: Path) -> None:
    if str(db_path).strip() == "":
        raise click.ClickException("db-path cannot be empty")

    _ensure_database(db_path)
    console.print(
        f"[green]OK[/green] Database initialized at: {db_path.resolve()}"
    )


@main.command("list-models")
def list_models() -> None:
    try:
        models = load_models_config()
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    table = Table(title="Configured Models")
    table.add_column("Alias", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Provider", style="green")
    table.add_column("OpenRouter Model ID", style="yellow")

    for model in models.values():
        table.add_row(
            model.alias,
            model.name,
            model.provider,
            model.model_id,
        )

    console.print(table)


@main.command("list-runs")
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    default=Path("./irbg.sqlite"),
    show_default=True,
    help="SQLite database file path.",
)
def list_runs(db_path: Path) -> None:
    _ensure_database(db_path)

    conn = connect(DbConfig(path=db_path))
    try:
        rows = list_benchmark_runs(conn)
    finally:
        conn.close()

    table = Table(title="Benchmark Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Mode", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Started At", style="blue")
    table.add_column("Completed At", style="blue")

    for row in rows:
        table.add_row(
            str(row["id"]),
            str(row["model_id"]),
            str(row["mode"]),
            str(row["status"]),
            str(row["started_at"]),
            str(row["completed_at"] or "-"),
        )

    console.print(table)


@main.command("list-variants")
@click.option(
    "--group",
    "group_name",
    required=True,
    help="Variant group from config/demographics.yaml",
)
def list_variants(group_name: str) -> None:
    try:
        variants = get_variant_group(group_name)
    except DemographicsError as exc:
        raise click.ClickException(str(exc)) from exc

    table = Table(title=f"Variants: {group_name}")
    table.add_column("Variant ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Gender", style="green")
    table.add_column("Nationality", style="yellow")
    table.add_column("Religion", style="blue")

    for variant in variants:
        table.add_row(
            variant.id,
            variant.name,
            variant.gender,
            variant.nationality,
            variant.religion,
        )

    console.print(table)


@main.command("ping-openrouter")
@click.option(
    "--model",
    "model_alias",
    required=True,
    help="Model alias from config/models.yaml",
)
@click.option(
    "--message",
    default="Reply with exactly one word: pong",
    show_default=True,
    help="Message to send to the model.",
)
def ping_openrouter(
    model_alias: str,
    message: str,
) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise click.ClickException(
            "OPENROUTER_API_KEY is not set. Add it to your .env file."
        )

    try:
        model = load_models_config()[model_alias]
    except KeyError as exc:
        raise click.ClickException(
            f"Unknown model alias: {model_alias}"
        ) from exc
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    client = OpenRouterClient(
        api_key=api_key,
        base_url=os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        ),
        app_name=os.getenv("OPENROUTER_APP_NAME", "IRBG"),
        site_url=os.getenv("OPENROUTER_SITE_URL"),
    )

    try:
        response = client.chat(
            model_id=model.model_id,
            system_prompt="You are a concise assistant.",
            user_prompt=message,
            temperature=model.temperature,
            max_tokens=model.max_tokens,
        )
    finally:
        client.close()

    if not response.success:
        raise click.ClickException(f"Provider request failed: {response.error}")

    console.print("[green]OpenRouter ping successful[/green]")
    console.print(f"[bold]Model:[/bold] {model.name}")
    console.print(f"[bold]Latency:[/bold] {response.latency_ms} ms")
    console.print(f"[bold]Tokens:[/bold] {response.total_tokens}")
    console.print(f"[bold]Response:[/bold] {response.text}")


@main.command("run-once")
@click.option(
    "--model",
    "model_alias",
    required=True,
    help="Model alias from config/models.yaml",
)
@click.option(
    "--scenario-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to a simple scenario JSON file.",
)
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    default=Path("./irbg.sqlite"),
    show_default=True,
    help="SQLite database file path.",
)
@click.option(
    "--mode",
    default="baseline",
    show_default=True,
    help="Execution mode label to store with the run.",
)
def run_once(
    model_alias: str,
    scenario_file: Path,
    db_path: Path,
    mode: str,
) -> None:
    _ensure_database(db_path)

    try:
        result = run_single_scenario(
            model_alias=model_alias,
            scenario_file=scenario_file,
            db_path=db_path,
            mode=mode,
        )
    except (
        ConfigError,
        RuntimeError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc

    if result.success:
        console.print("[green]Scenario run completed successfully[/green]")
        console.print(f"[bold]Run ID:[/bold] {result.run_id}")
        console.print(f"[bold]Response ID:[/bold] {result.response_id}")
        console.print(f"[bold]Model Alias:[/bold] {result.model_alias}")
        console.print(f"[bold]Scenario ID:[/bold] {result.scenario_id}")
    else:
        console.print("[red]Scenario run failed[/red]")
        console.print(f"[bold]Run ID:[/bold] {result.run_id}")
        console.print(f"[bold]Response ID:[/bold] {result.response_id}")
        console.print(f"[bold]Error:[/bold] {result.error}")
        raise click.ClickException("Single scenario run failed.")


@main.command("render-template")
@click.option(
    "--scenario-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to a template scenario JSON file.",
)
@click.option(
    "--variant-id",
    required=True,
    help="Variant ID from demographics.yaml",
)
@click.option(
    "--mode",
    default="baseline",
    show_default=True,
    help="Render mode for the prompt.",
)
def render_template(
    scenario_file: Path,
    variant_id: str,
    mode: str,
) -> None:
    try:
        template = load_scenario_template(scenario_file)
        rendered = generate_single_prompt_for_variant(
            template,
            variant_id=variant_id,
            mode=mode,
        )
    except (
        ScenarioTemplateLoadError,
        VariantGenerationError,
        DemographicsError,
        PromptBuildError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc

    metadata = Table(show_header=False, box=None)
    metadata.add_row("Scenario ID", rendered.scenario_id)
    metadata.add_row("Variant ID", rendered.variant_id or "-")
    metadata.add_row("Mode", rendered.mode)
    metadata.add_row("Jurisdiction", rendered.jurisdiction or "-")
    metadata.add_row("Category", rendered.category)

    console.print(Panel.fit(metadata, title="Rendered Prompt Metadata"))
    console.print(
        Panel(
            rendered.system_prompt,
            title="System Prompt",
            expand=False,
        )
    )
    console.print(
        Panel(
            rendered.user_prompt,
            title="User Prompt",
            expand=False,
        )
    )


@main.command("run-template-variant")
@click.option(
    "--model",
    "model_alias",
    required=True,
    help="Model alias from config/models.yaml",
)
@click.option(
    "--scenario-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to a template scenario JSON file.",
)
@click.option(
    "--variant-id",
    required=True,
    help="Variant ID from demographics.yaml",
)
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    default=Path("./irbg.sqlite"),
    show_default=True,
    help="SQLite database file path.",
)
@click.option(
    "--mode",
    default="baseline",
    show_default=True,
    help="Execution mode label to store with the run.",
)
def run_template_variant(
    model_alias: str,
    scenario_file: Path,
    variant_id: str,
    db_path: Path,
    mode: str,
) -> None:
    _ensure_database(db_path)

    try:
        result = run_single_template_variant(
            model_alias=model_alias,
            scenario_file=scenario_file,
            variant_id=variant_id,
            db_path=db_path,
            mode=mode,
        )
    except (
        ConfigError,
        RuntimeError,
        ScenarioTemplateLoadError,
        VariantGenerationError,
        DemographicsError,
        PromptBuildError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc

    if result.success:
        console.print(
            "[green]Template variant run completed successfully[/green]"
        )
        console.print(f"[bold]Run ID:[/bold] {result.run_id}")
        console.print(f"[bold]Response ID:[/bold] {result.response_id}")
        console.print(f"[bold]Model Alias:[/bold] {result.model_alias}")
        console.print(f"[bold]Scenario ID:[/bold] {result.scenario_id}")
    else:
        console.print("[red]Template variant run failed[/red]")
        console.print(f"[bold]Run ID:[/bold] {result.run_id}")
        console.print(f"[bold]Response ID:[/bold] {result.response_id}")
        console.print(f"[bold]Error:[/bold] {result.error}")
        raise click.ClickException("Template variant run failed.")


@main.command("run-template-group")
@click.option(
    "--model",
    "model_alias",
    required=True,
    help="Model alias from config/models.yaml",
)
@click.option(
    "--scenario-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to a template scenario JSON file.",
)
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    default=Path("./irbg.sqlite"),
    show_default=True,
    help="SQLite database file path.",
)
@click.option(
    "--mode",
    default="baseline",
    show_default=True,
    help="Execution mode label to store with the run.",
)
def run_template_group(
    model_alias: str,
    scenario_file: Path,
    db_path: Path,
    mode: str,
) -> None:
    _ensure_database(db_path)

    try:
        result = run_all_template_variants(
            model_alias=model_alias,
            scenario_file=scenario_file,
            db_path=db_path,
            mode=mode,
        )
    except (
        ConfigError,
        RuntimeError,
        ScenarioTemplateLoadError,
        VariantGenerationError,
        DemographicsError,
        PromptBuildError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc

    console.print("[green]Template group run finished[/green]")
    console.print(f"[bold]Run ID:[/bold] {result.run_id}")
    console.print(f"[bold]Model Alias:[/bold] {result.model_alias}")
    console.print(f"[bold]Scenario ID:[/bold] {result.scenario_id}")
    console.print(f"[bold]Mode:[/bold] {result.mode}")
    console.print(f"[bold]Total:[/bold] {result.total_count}")
    console.print(f"[bold]Succeeded:[/bold] {result.success_count}")
    console.print(f"[bold]Failed:[/bold] {result.failure_count}")


@main.command("run-template-folder")
@click.option(
    "--model",
    "model_alias",
    required=True,
    help="Model alias from config/models.yaml",
)
@click.option(
    "--scenario-folder",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Folder containing template scenario JSON files.",
)
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    default=Path("./irbg.sqlite"),
    show_default=True,
    help="SQLite database file path.",
)
@click.option(
    "--mode",
    default="baseline",
    show_default=True,
    help="Execution mode label to store with the run.",
)
def run_template_folder_cmd(
    model_alias: str,
    scenario_folder: Path,
    db_path: Path,
    mode: str,
) -> None:
    _ensure_database(db_path)

    try:
        result = run_template_folder(
            model_alias=model_alias,
            folder_path=scenario_folder,
            db_path=db_path,
            mode=mode,
        )
    except (
        ConfigError,
        RuntimeError,
        ScenarioDiscoveryError,
        ScenarioTemplateLoadError,
        VariantGenerationError,
        DemographicsError,
        PromptBuildError,
        ValueError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc

    console.print("[green]Template folder run finished[/green]")
    console.print(f"[bold]Run ID:[/bold] {result.run_id}")
    console.print(f"[bold]Model Alias:[/bold] {result.model_alias}")
    console.print(f"[bold]Folder:[/bold] {result.folder_path}")
    console.print(f"[bold]Mode:[/bold] {result.mode}")
    console.print(f"[bold]Scenarios:[/bold] {result.scenario_count}")
    console.print(
        f"[bold]Total Prompt Count:[/bold] {result.total_prompt_count}"
    )
    console.print(f"[bold]Succeeded:[/bold] {result.success_count}")
    console.print(f"[bold]Failed:[/bold] {result.failure_count}")


@main.command("score-p1-run")
@click.option(
    "--run-id",
    required=True,
    help="Benchmark run ID to score.",
)
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    default=Path("./irbg.sqlite"),
    show_default=True,
    help="SQLite database file path.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional path to save the JSON score report.",
)
def score_p1_run_cmd(
    run_id: str,
    db_path: Path,
    output: Path | None,
) -> None:
    _ensure_database(db_path)

    try:
        result = score_p1_run(
            db_path=db_path,
            run_id=run_id,
        )
    except P1ScoringError as exc:
        raise click.ClickException(str(exc)) from exc

    summary = Table(title="P1 Run Score")
    summary.add_column("Field", style="cyan")
    summary.add_column("Value", style="magenta")
    summary.add_row("Run ID", result.run_id)
    summary.add_row("Model", result.model_alias)
    summary.add_row("Mode", result.mode)
    summary.add_row("Scenario Count", str(result.scenario_count))
    summary.add_row("Overall Score", f"{result.overall_score:.2f}")

    console.print(summary)

    detail = Table(title="Scenario Breakdown")
    detail.add_column("Scenario ID", style="cyan")
    detail.add_column("Category", style="green")
    detail.add_column("Decision", style="yellow")
    detail.add_column("Length", style="yellow")
    detail.add_column("Sentiment", style="yellow")
    detail.add_column("Total", style="magenta")
    detail.add_column("Majority", style="blue")
    detail.add_column("Outliers", style="red")

    for item in result.scenarios:
        outliers = ", ".join(item.outlier_variants) or "-"
        detail.add_row(
            item.scenario_id,
            item.category,
            f"{item.decision_score:.2f}",
            f"{item.length_score:.2f}",
            f"{item.sentiment_score:.2f}",
            f"{item.total_score:.2f}",
            item.majority_decision,
            outliers,
        )

    console.print(detail)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(asdict(result), indent=2),
        )
        console.print(
            f"[green]Saved JSON score report to[/green] {output.resolve()}"
        )


if __name__ == "__main__":
    main()
