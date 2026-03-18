from __future__ import annotations

import sqlite3


def create_tables(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON;")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            provider TEXT NOT NULL,
            model_id TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            mode TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            config_snapshot TEXT,
            FOREIGN KEY (model_id) REFERENCES models (id)
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scenarios (
            id TEXT PRIMARY KEY,
            pillar TEXT NOT NULL,
            category TEXT NOT NULL,
            jurisdiction TEXT,
            difficulty TEXT,
            created_at TEXT NOT NULL
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS responses (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            scenario_id TEXT NOT NULL,
            variant_id TEXT,
            mode TEXT NOT NULL,
            turn_number INTEGER NOT NULL DEFAULT 1,
            system_prompt_sent TEXT NOT NULL,
            user_prompt_sent TEXT NOT NULL,
            raw_response TEXT,
            response_tokens INTEGER,
            latency_ms INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES benchmark_runs (id),
            FOREIGN KEY (scenario_id) REFERENCES scenarios (id)
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scores (
            id TEXT PRIMARY KEY,
            response_id TEXT NOT NULL,
            scorer_type TEXT NOT NULL,
            total_score REAL NOT NULL,
            breakdown_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (response_id) REFERENCES responses (id)
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pillar_scores (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            pillar TEXT NOT NULL,
            score REAL NOT NULL,
            breakdown_json TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            UNIQUE (run_id, pillar),
            FOREIGN KEY (run_id) REFERENCES benchmark_runs (id)
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS irbg_scores (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL UNIQUE,
            composite_score REAL NOT NULL,
            grade TEXT NOT NULL,
            breakdown_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES benchmark_runs (id)
        );
        """
    )

    conn.commit()
