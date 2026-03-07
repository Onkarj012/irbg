from __future__ import annotations

import os
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from irbg.config import ConfigError, load_models_config
from irbg.db.operations import DbConfig, connect
from irbg.db.schema import create_tables
from irbg.engine.provider import OpenRouterClient
from irbg.engine.runner import run_single_scenario

console = Console()


@click.group()
def main() -> None:
    """
    IRBG (Institutional Readiness & Bias Benchmark for Governance)

    CLI for database setup, provider checks, and benchmark execution.
    """
    load_dotenv()


@main.command("init-db")
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    default=Path("./irbg.sqlite"),
    show_default=True,
    help="SQLite database file path.",
)
def init_db(db_path: Path) -> None:
    """
    Create the SQLite database and required tables.
    """
    if str(db_path).strip() == "":
        raise click.ClickException("db-path cannot be empty")

    db = DbConfig(path=db_path)
    conn = connect(db)

    try:
        create_tables(conn)
    finally:
        conn.close()

    console.print(
        f"[green]OK[/green] Database initialized at: {db_path.resolve()}"
    )


@main.command("list-models")
def list_models() -> None:
    """
    Display configured model aliases from config/models.yaml.
    """
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
    """
    Send a simple test prompt to OpenRouter and print the response.
    """
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
    help="Path to a scenario JSON file.",
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
    """
    Run a single scenario end-to-end and store the response in SQLite.
    """
    if not db_path.exists():
        db = DbConfig(path=db_path)
        conn = connect(db)
        try:
            create_tables(conn)
        finally:
            conn.close()

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


if __name__ == "__main__":
    main()
