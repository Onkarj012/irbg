from __future__ import annotations

from irbg.analysis.reporting import build_run_report
from irbg.db.operations import (
    DbConfig,
    connect,
    create_benchmark_run,
    insert_response,
    upsert_model,
    upsert_pillar_score,
    upsert_scenario_record,
)
from irbg.db.schema import create_tables


def test_build_run_report(tmp_path) -> None:
    db_path = tmp_path / "report.sqlite"
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
        status="completed",
        config_snapshot="{}",
    )

    insert_response(
        conn,
        run_id=run_id,
        scenario_id="P1-US-BAIL-001",
        variant_id="us_white_male",
        mode="baseline",
        turn_number=1,
        system_prompt_sent="system",
        user_prompt_sent="user",
        raw_response="Grant bail with standard conditions.",
        response_tokens=21,
        latency_ms=123,
    )

    upsert_pillar_score(
        conn,
        run_id=run_id,
        pillar="p1_demographic_consistency",
        score=91.2,
        breakdown_json="{}",
        notes="test",
    )
    conn.close()

    report = build_run_report(
        db_path=db_path,
        run_id=run_id,
    )

    assert report.run_id == run_id
    assert report.model_alias == "gpt-4o"
    assert report.response_count == 1
    assert report.scenario_count == 1
    assert report.average_latency_ms == 123.0
    assert report.average_tokens == 21.0
    assert report.pillar_scores["p1_demographic_consistency"] == 91.2
