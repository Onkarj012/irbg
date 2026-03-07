from __future__ import annotations

from pathlib import Path

from irbg.db.operations import (
    DbConfig,
    connect,
    create_benchmark_run,
    insert_response,
    upsert_model,
    upsert_scenario,
)
from irbg.db.schema import create_tables
from irbg.scenarios.models import Scenario


def test_insert_model_run_and_response(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite"
    conn = connect(DbConfig(path=db_path))
    create_tables(conn)

    upsert_model(
        conn,
        id="gpt-4o",
        name="GPT-4o",
        provider="openrouter",
        model_id="openai/gpt-4o",
    )

    scenario = Scenario(
        id="SCENARIO-001",
        pillar="smoke",
        category="demo",
        jurisdiction="general",
        difficulty="easy",
        system_prompt="You are helpful.",
        user_prompt="Say hello.",
    )
    upsert_scenario(conn, scenario=scenario)

    run_id = create_benchmark_run(
        conn,
        model_id="gpt-4o",
        mode="baseline",
        status="running",
        config_snapshot="{}",
    )

    response_id = insert_response(
        conn,
        run_id=run_id,
        scenario_id="SCENARIO-001",
        variant_id=None,
        mode="baseline",
        turn_number=1,
        system_prompt_sent="You are helpful.",
        user_prompt_sent="Say hello.",
        raw_response="Hello",
        response_tokens=5,
        latency_ms=100,
    )

    run_row = conn.execute(
        "SELECT * FROM benchmark_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    response_row = conn.execute(
        "SELECT * FROM responses WHERE id = ?",
        (response_id,),
    ).fetchone()

    assert run_row is not None
    assert response_row is not None
    assert response_row["raw_response"] == "Hello"

    conn.close()
