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


def upsert_pillar_score(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    pillar: str,
    score: float,
    breakdown_json: str | None,
    notes: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO pillar_scores (
            id,
            run_id,
            pillar,
            score,
            breakdown_json,
            notes,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, pillar) DO UPDATE SET
            score = excluded.score,
            breakdown_json = excluded.breakdown_json,
            notes = excluded.notes,
            created_at = excluded.created_at;
        """,
        (
            new_id(),
            run_id,
            pillar,
            score,
            breakdown_json,
            notes,
            now_utc_iso(),
        ),
    )
    conn.commit()


def upsert_irbg_score(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    composite_score: float,
    grade: str,
    breakdown_json: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO irbg_scores (
            id,
            run_id,
            composite_score,
            grade,
            breakdown_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            composite_score = excluded.composite_score,
            grade = excluded.grade,
            breakdown_json = excluded.breakdown_json,
            created_at = excluded.created_at;
        """,
        (
            new_id(),
            run_id,
            composite_score,
            grade,
            breakdown_json,
            now_utc_iso(),
        ),
    )
    conn.commit()


def list_benchmark_runs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT
            id,
            model_id,
            mode,
            status,
            started_at,
            completed_at
        FROM benchmark_runs
        ORDER BY started_at DESC;
        """
    ).fetchall()

    return list(rows)


def get_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT
            id,
            model_id,
            mode,
            status,
            started_at,
            completed_at,
            config_snapshot
        FROM benchmark_runs
        WHERE id = ?;
        """,
        (run_id,),
    ).fetchone()


def get_responses_for_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
) -> list[sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT
            r.id,
            r.run_id,
            r.scenario_id,
            r.variant_id,
            r.mode,
            r.turn_number,
            r.system_prompt_sent,
            r.user_prompt_sent,
            r.raw_response,
            r.response_tokens,
            r.latency_ms,
            r.created_at,
            s.pillar,
            s.category,
            s.jurisdiction,
            s.difficulty
        FROM responses r
        JOIN scenarios s
            ON r.scenario_id = s.id
        WHERE r.run_id = ?
        ORDER BY r.scenario_id, r.variant_id, r.created_at;
        """,
        (run_id,),
    ).fetchall()

    return list(rows)


def get_pillar_score(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    pillar: str,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT
            id,
            run_id,
            pillar,
            score,
            breakdown_json,
            notes,
            created_at
        FROM pillar_scores
        WHERE run_id = ? AND pillar = ?;
        """,
        (run_id, pillar),
    ).fetchone()


def get_all_pillar_scores(
    conn: sqlite3.Connection,
    *,
    run_id: str,
) -> list[sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT
            id,
            run_id,
            pillar,
            score,
            breakdown_json,
            notes,
            created_at
        FROM pillar_scores
        WHERE run_id = ?
        ORDER BY pillar;
        """,
        (run_id,),
    ).fetchall()

    return list(rows)


def get_irbg_score(
    conn: sqlite3.Connection,
    *,
    run_id: str,
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT
            id,
            run_id,
            composite_score,
            grade,
            breakdown_json,
            created_at
        FROM irbg_scores
        WHERE run_id = ?;
        """,
        (run_id,),
    ).fetchone()
