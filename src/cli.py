from __future__ import annotations

from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console

from irbg.db.operations import DbConfig, connect
from irbg.db.schema import create_tables

console = Console()


@click.group()
def main() -> None:
    """
    IRBG (Institutional Readiness & Bias Benchmark for Governance)

    Use this CLI to initialize the database and (later) run benchmarks.
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

    if not str(db_path).strip():
        raise click.ClickException("db-path is required")

    db = DbConfig(path=db_path)
    conn = connect(db)

    try:
        create_tables(conn)
    finally:
        conn.close()

    console.print(
        f"[green]OK[/green] Database initialized at: {db_path.resolve()}"
    )


if __name__ == "__main__":
    main()
