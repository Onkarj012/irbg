from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from irbg.scenarios.models import Scenario


@dataclass(frozen=True)
class DbConfig:
    path: Path


def connect(db: DbConfig) -> sqlite3.Connection:
    db.path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db.path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def now_utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_id() -> str:
    return uuid4().hex


def upsert_model(
    conn: sqlite3.Connection,
    *,
    id: str,
    name: str,
    provider: str,
    model_id: str,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO models (
            id,
            name,
            provider,
            model_id,
            created_at
        )
        VALUES (?, ?, ?, ?, ?);
        """,
        (id, name, provider, model_id, now_utc_iso()),
    )
    conn.commit()


def create_benchmark_run(
    conn: sqlite3.Connection,
    *,
    model_id: str,
    mode: str,
    status: str,
    config_snapshot: str | None = None,
) -> str:
    run_id = new_id()

    conn.execute(
        """
        INSERT INTO benchmark_runs (
            id,
            model_id,
            mode,
            status,
            started_at,
            completed_at,
            config_snapshot
        )
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        (
            run_id,
            model_id,
            mode,
            status,
            now_utc_iso(),
            None,
            config_snapshot,
        ),
    )
    conn.commit()
    return run_id


def mark_benchmark_run_completed(
    conn: sqlite3.Connection,
    *,
    run_id: str,
) -> None:
    conn.execute(
        """
        UPDATE benchmark_runs
        SET status = ?, completed_at = ?
        WHERE id = ?;
        """,
        ("completed", now_utc_iso(), run_id),
    )
    conn.commit()


def mark_benchmark_run_failed(
    conn: sqlite3.Connection,
    *,
    run_id: str,
) -> None:
    conn.execute(
        """
        UPDATE benchmark_runs
        SET status = ?, completed_at = ?
        WHERE id = ?;
        """,
        ("failed", now_utc_iso(), run_id),
    )
    conn.commit()


def upsert_scenario_record(
    conn: sqlite3.Connection,
    *,
    id: str,
    pillar: str,
    category: str,
    jurisdiction: str | None,
    difficulty: str | None,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO scenarios (
            id,
            pillar,
            category,
            jurisdiction,
            difficulty,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        (
            id,
            pillar,
            category,
            jurisdiction,
            difficulty,
            now_utc_iso(),
        ),
    )
    conn.commit()


def upsert_scenario(
    conn: sqlite3.Connection,
    *,
    scenario: Scenario,
) -> None:
    upsert_scenario_record(
        conn,
        id=scenario.id,
        pillar=scenario.pillar,
        category=scenario.category,
        jurisdiction=scenario.jurisdiction,
        difficulty=scenario.difficulty,
    )


def insert_response(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    scenario_id: str,
    variant_id: str | None,
    mode: str,
    turn_number: int,
    system_prompt_sent: str,
    user_prompt_sent: str,
    raw_response: str | None,
    response_tokens: int | None,
    latency_ms: int | None,
) -> str:
    response_id = new_id()

    conn.execute(
        """
        INSERT INTO responses (
            id,
            run_id,
            scenario_id,
            variant_id,
            mode,
            turn_number,
            system_prompt_sent,
            user_prompt_sent,
            raw_response,
            response_tokens,
            latency_ms,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            response_id,
            run_id,
            scenario_id,
            variant_id,
            mode,
            turn_number,
            system_prompt_sent,
            user_prompt_sent,
            raw_response,
            response_tokens,
            latency_ms,
            now_utc_iso(),
        ),
    )
    conn.commit()
    return response_id
