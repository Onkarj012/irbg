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

from irbg.analysis.aggregate import (
    AggregateScoreError,
    aggregate_run_score,
)
from irbg.analysis.compare import compare_runs
from irbg.analysis.reporting import (
    RunReportError,
    build_run_report,
    write_run_report_json,
    write_run_report_markdown,
)
from irbg.analysis.visualize import (
    VisualizationError,
    generate_latency_chart,
    generate_run_summary_chart,
)
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
)
def init_db(db_path: Path) -> None:
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
    table.add_column("Alias")
    table.add_column("Name")
    table.add_column("Provider")
    table.add_column("OpenRouter Model ID")

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
)
def list_runs(db_path: Path) -> None:
    _ensure_database(db_path)

    conn = connect(DbConfig(path=db_path))
    try:
        rows = list_benchmark_runs(conn)
    finally:
        conn.close()

    table = Table(title="Benchmark Runs")
    table.add_column("Run ID")
    table.add_column("Model")
    table.add_column("Mode")
    table.add_column("Status")
    table.add_column("Started At")
    table.add_column("Completed At")

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
@click.option("--group", "group_name", required=True)
def list_variants(group_name: str) -> None:
    try:
        variants = get_variant_group(group_name)
    except DemographicsError as exc:
        raise click.ClickException(str(exc)) from exc

    table = Table(title=f"Variants: {group_name}")
    table.add_column("Variant ID")
    table.add_column("Name")
    table.add_column("Gender")
    table.add_column("Nationality")
    table.add_column("Religion")

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
@click.option("--model", "model_alias", required=True)
@click.option(
    "--message",
    default="Reply with exactly one word: pong",
    show_default=True,
)
def ping_openrouter(model_alias: str, message: str) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise click.ClickException("OPENROUTER_API_KEY is not set.")

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
        raise click.ClickException(response.error or "Provider error")

    console.print("[green]OpenRouter ping successful[/green]")
    console.print(f"Model: {model.name}")
    console.print(f"Latency: {response.latency_ms} ms")
    console.print(f"Tokens: {response.total_tokens}")
    console.print(f"Response: {response.text}")


@main.command("render-template")
@click.option(
    "--scenario-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option("--variant-id", required=True)
@click.option("--mode", default="baseline", show_default=True)
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
    console.print(Panel(rendered.system_prompt, title="System Prompt"))
    console.print(Panel(rendered.user_prompt, title="User Prompt"))


@main.command("run-once")
@click.option("--model", "model_alias", required=True)
@click.option(
    "--scenario-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    default=Path("./irbg.sqlite"),
    show_default=True,
)
@click.option("--mode", default="baseline", show_default=True)
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
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if not result.success:
        raise click.ClickException(result.error or "Run failed")

    console.print("[green]Scenario run completed successfully[/green]")
    console.print(f"Run ID: {result.run_id}")
    console.print(f"Response ID: {result.response_id}")
    console.print(f"Model Alias: {result.model_alias}")
    console.print(f"Scenario ID: {result.scenario_id}")


@main.command("run-template-variant")
@click.option("--model", "model_alias", required=True)
@click.option(
    "--scenario-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option("--variant-id", required=True)
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    default=Path("./irbg.sqlite"),
    show_default=True,
)
@click.option("--mode", default="baseline", show_default=True)
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
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    if not result.success:
        raise click.ClickException(result.error or "Run failed")

    console.print("[green]Template variant run completed[/green]")
    console.print(f"Run ID: {result.run_id}")
    console.print(f"Response ID: {result.response_id}")
    console.print(f"Model Alias: {result.model_alias}")
    console.print(f"Scenario ID: {result.scenario_id}")


@main.command("run-template-group")
@click.option("--model", "model_alias", required=True)
@click.option(
    "--scenario-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    default=Path("./irbg.sqlite"),
    show_default=True,
)
@click.option("--mode", default="baseline", show_default=True)
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
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    console.print("[green]Template group run finished[/green]")
    console.print(f"Run ID: {result.run_id}")
    console.print(f"Model Alias: {result.model_alias}")
    console.print(f"Scenario ID: {result.scenario_id}")
    console.print(f"Mode: {result.mode}")
    console.print(f"Total: {result.total_count}")
    console.print(f"Succeeded: {result.success_count}")
    console.print(f"Failed: {result.failure_count}")


@main.command("run-template-folder")
@click.option("--model", "model_alias", required=True)
@click.option(
    "--scenario-folder",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    default=Path("./irbg.sqlite"),
    show_default=True,
)
@click.option("--mode", default="baseline", show_default=True)
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
    console.print(f"Run ID: {result.run_id}")
    console.print(f"Model Alias: {result.model_alias}")
    console.print(f"Folder: {result.folder_path}")
    console.print(f"Mode: {result.mode}")
    console.print(f"Scenarios: {result.scenario_count}")
    console.print(f"Total Prompt Count: {result.total_prompt_count}")
    console.print(f"Succeeded: {result.success_count}")
    console.print(f"Failed: {result.failure_count}")


@main.command("score-p1-run")
@click.option("--run-id", required=True)
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    default=Path("./irbg.sqlite"),
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
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
    summary.add_column("Field")
    summary.add_column("Value")
    summary.add_row("Run ID", result.run_id)
    summary.add_row("Model", result.model_alias)
    summary.add_row("Mode", result.mode)
    summary.add_row("Scenario Count", str(result.scenario_count))
    summary.add_row("Overall Score", f"{result.overall_score:.2f}")
    console.print(summary)

    detail = Table(title="Scenario Breakdown")
    detail.add_column("Scenario ID")
    detail.add_column("Category")
    detail.add_column("Decision")
    detail.add_column("Length")
    detail.add_column("Sentiment")
    detail.add_column("Total")
    detail.add_column("Majority")
    detail.add_column("Outliers")

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
        output.write_text(json.dumps(asdict(result), indent=2))
        console.print(f"Saved JSON score report to {output.resolve()}")


@main.command("aggregate-run")
@click.option("--run-id", required=True)
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    default=Path("./irbg.sqlite"),
    show_default=True,
)
def aggregate_run_cmd(
    run_id: str,
    db_path: Path,
) -> None:
    _ensure_database(db_path)

    try:
        result = aggregate_run_score(db_path=db_path, run_id=run_id)
    except AggregateScoreError as exc:
        raise click.ClickException(str(exc)) from exc

    table = Table(title="Aggregated IRBG Score")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Run ID", result.run_id)
    table.add_row("Model", result.model_alias)
    table.add_row("Mode", result.mode)
    table.add_row("Composite Score", f"{result.composite_score:.2f}")
    table.add_row("Grade", result.grade)

    for pillar, score in sorted(result.pillar_scores.items()):
        table.add_row(pillar, f"{score:.2f}")

    console.print(table)


@main.command("show-run")
@click.option("--run-id", required=True)
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    default=Path("./irbg.sqlite"),
    show_default=True,
)
def show_run_cmd(
    run_id: str,
    db_path: Path,
) -> None:
    _ensure_database(db_path)

    try:
        report = build_run_report(db_path=db_path, run_id=run_id)
    except RunReportError as exc:
        raise click.ClickException(str(exc)) from exc

    table = Table(title="Run Summary")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Run ID", report.run_id)
    table.add_row("Model", report.model_alias)
    table.add_row("Mode", report.mode)
    table.add_row("Status", report.status)
    table.add_row("Response Count", str(report.response_count))
    table.add_row("Scenario Count", str(report.scenario_count))
    table.add_row(
        "Average Latency (ms)",
        f"{report.average_latency_ms:.2f}",
    )
    table.add_row("Average Tokens", f"{report.average_tokens:.2f}")
    table.add_row(
        "Composite Score",
        str(report.composite_score)
        if report.composite_score is not None
        else "-",
    )
    table.add_row("Grade", report.grade or "-")

    console.print(table)

    if report.pillar_scores:
        pillar_table = Table(title="Pillar Scores")
        pillar_table.add_column("Pillar")
        pillar_table.add_column("Score")
        for pillar, score in sorted(report.pillar_scores.items()):
            pillar_table.add_row(pillar, f"{score:.2f}")
        console.print(pillar_table)


@main.command("report-run")
@click.option("--run-id", required=True)
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    default=Path("./irbg.sqlite"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./reports"),
    show_default=True,
)
def report_run_cmd(
    run_id: str,
    db_path: Path,
    output_dir: Path,
) -> None:
    _ensure_database(db_path)

    try:
        report = build_run_report(db_path=db_path, run_id=run_id)
    except RunReportError as exc:
        raise click.ClickException(str(exc)) from exc

    json_path = output_dir / f"{run_id}_report.json"
    md_path = output_dir / f"{run_id}_report.md"
    pillar_chart_path = output_dir / f"{run_id}_pillar_scores.png"
    latency_chart_path = output_dir / f"{run_id}_latency.png"

    write_run_report_json(report=report, output_path=json_path)
    write_run_report_markdown(report=report, output_path=md_path)

    try:
        generate_run_summary_chart(
            db_path=db_path,
            run_id=run_id,
            output_path=pillar_chart_path,
        )
    except VisualizationError as exc:
        console.print(f"[yellow]Chart skipped:[/yellow] {exc}")

    try:
        generate_latency_chart(
            db_path=db_path,
            run_id=run_id,
            output_path=latency_chart_path,
        )
    except VisualizationError as exc:
        console.print(f"[yellow]Chart skipped:[/yellow] {exc}")

    console.print("[green]Run report generated[/green]")
    console.print(f"JSON: {json_path.resolve()}")
    console.print(f"Markdown: {md_path.resolve()}")
    console.print(f"Pillar chart: {pillar_chart_path.resolve()}")
    console.print(f"Latency chart: {latency_chart_path.resolve()}")


@main.command("compare-runs")
@click.option("--left-run-id", required=True)
@click.option("--right-run-id", required=True)
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    default=Path("./irbg.sqlite"),
    show_default=True,
)
def compare_runs_cmd(
    left_run_id: str,
    right_run_id: str,
    db_path: Path,
) -> None:
    _ensure_database(db_path)

    comparison = compare_runs(
        db_path=db_path,
        left_run_id=left_run_id,
        right_run_id=right_run_id,
    )

    table = Table(title="Run Comparison")
    table.add_column("Field")
    table.add_column("Left")
    table.add_column("Right")

    table.add_row("Run ID", comparison.left_run_id, comparison.right_run_id)
    table.add_row("Model", comparison.left_model, comparison.right_model)
    table.add_row(
        "Composite Score",
        str(comparison.left_score),
        str(comparison.right_score),
    )
    table.add_row(
        "Grade",
        str(comparison.left_grade),
        str(comparison.right_grade),
    )

    console.print(table)

    delta_text = (
        str(comparison.score_delta)
        if comparison.score_delta is not None
        else "N/A"
    )
    console.print(f"Score delta (left - right): {delta_text}")


if __name__ == "__main__":
    main()
