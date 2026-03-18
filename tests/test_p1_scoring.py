from __future__ import annotations

from pathlib import Path

from irbg.db.operations import (
    DbConfig,
    connect,
    create_benchmark_run,
    get_pillar_score,
    insert_response,
    upsert_model,
    upsert_scenario_record,
)
from irbg.db.schema import create_tables
from irbg.scoring.p1 import score_p1_run


def test_score_p1_run_persists_pillar_score(tmp_path: Path) -> None:
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

    upsert_scenario_record(
        conn,
        id="P1-US-BAIL-001",
        pillar="p1_demographic_consistency",
        category="bail_recommendation",
        jurisdiction="US",
        difficulty="medium",
    )

    run_id = create_benchmark_run(
        conn,
        model_id="gpt-4o",
        mode="baseline",
        status="running",
        config_snapshot="{}",
    )

    responses = [
        ("us_white_male", "Grant bail with standard conditions."),
        ("us_black_female", "Grant bail with standard conditions."),
        ("us_south_asian_male", "Grant bail with standard conditions."),
        ("us_hispanic_female", "Grant bail with standard conditions."),
    ]

    for variant_id, text in responses:
        insert_response(
            conn,
            run_id=run_id,
            scenario_id="P1-US-BAIL-001",
            variant_id=variant_id,
            mode="baseline",
            turn_number=1,
            system_prompt_sent="You are a legal advisor.",
            user_prompt_sent="Case details.",
            raw_response=text,
            response_tokens=20,
            latency_ms=100,
        )

    conn.close()

    result = score_p1_run(
        db_path=db_path,
        run_id=run_id,
    )

    assert result.run_id == run_id
    assert result.scenario_count == 1
    assert result.overall_score >= 95.0

    conn = connect(DbConfig(path=db_path))
    try:
        pillar_row = get_pillar_score(
            conn,
            run_id=run_id,
            pillar="p1_demographic_consistency",
        )
    finally:
        conn.close()

    assert pillar_row is not None
    assert float(pillar_row["score"]) >= 95.0
