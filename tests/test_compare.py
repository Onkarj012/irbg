from __future__ import annotations

from irbg.analysis.aggregate import aggregate_run_score
from irbg.analysis.compare import compare_runs
from irbg.db.operations import (
    DbConfig,
    connect,
    create_benchmark_run,
    upsert_model,
    upsert_pillar_score,
)
from irbg.db.schema import create_tables


def test_compare_runs(tmp_path) -> None:
    db_path = tmp_path / "compare.sqlite"
    conn = connect(DbConfig(path=db_path))
    create_tables(conn)

    upsert_model(
        conn,
        id="model_a",
        name="Model A",
        provider="openrouter",
        model_id="provider/model-a",
    )
    upsert_model(
        conn,
        id="model_b",
        name="Model B",
        provider="openrouter",
        model_id="provider/model-b",
    )

    run_a = create_benchmark_run(
        conn,
        model_id="model_a",
        mode="baseline",
        status="completed",
        config_snapshot="{}",
    )
    run_b = create_benchmark_run(
        conn,
        model_id="model_b",
        mode="baseline",
        status="completed",
        config_snapshot="{}",
    )

    upsert_pillar_score(
        conn,
        run_id=run_a,
        pillar="p1_demographic_consistency",
        score=85.0,
        breakdown_json="{}",
        notes="a",
    )
    upsert_pillar_score(
        conn,
        run_id=run_b,
        pillar="p1_demographic_consistency",
        score=75.0,
        breakdown_json="{}",
        notes="b",
    )
    conn.close()

    aggregate_run_score(db_path=db_path, run_id=run_a)
    aggregate_run_score(db_path=db_path, run_id=run_b)

    result = compare_runs(
        db_path=db_path,
        left_run_id=run_a,
        right_run_id=run_b,
    )

    assert result.left_score == 85.0
    assert result.right_score == 75.0
    assert result.score_delta == 10.0
