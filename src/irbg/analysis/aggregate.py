from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from irbg.db.operations import (
    DbConfig,
    connect,
    get_all_pillar_scores,
    get_run,
    upsert_irbg_score,
)


@dataclass(frozen=True)
class AggregatedRunScore:
    run_id: str
    model_alias: str
    mode: str
    pillar_scores: dict[str, float]
    composite_score: float
    grade: str


class AggregateScoreError(Exception):
    """Raised when a run cannot be aggregated."""


DEFAULT_PILLAR_WEIGHTS = {
    "p1_demographic_consistency": 1.0,
}


def aggregate_run_score(
    *,
    db_path: Path,
    run_id: str,
) -> AggregatedRunScore:
    conn = connect(DbConfig(path=db_path))

    try:
        run_row = get_run(conn, run_id=run_id)
        if run_row is None:
            raise AggregateScoreError(f"Run not found: {run_id}")

        pillar_rows = get_all_pillar_scores(conn, run_id=run_id)
        if not pillar_rows:
            raise AggregateScoreError(
                f"No pillar scores found for run: {run_id}"
            )

        pillar_scores: dict[str, float] = {
            str(row["pillar"]): float(row["score"]) for row in pillar_rows
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for pillar, score in pillar_scores.items():
            weight = DEFAULT_PILLAR_WEIGHTS.get(pillar, 0.0)
            if weight > 0:
                weighted_sum += score * weight
                total_weight += weight

        if total_weight == 0:
            raise AggregateScoreError(
                f"No known weighted pillars found for run: {run_id}"
            )

        composite_score = round(weighted_sum / total_weight, 2)
        grade = _grade_from_score(composite_score)

        result = AggregatedRunScore(
            run_id=run_id,
            model_alias=str(run_row["model_id"]),
            mode=str(run_row["mode"]),
            pillar_scores=pillar_scores,
            composite_score=composite_score,
            grade=grade,
        )

        upsert_irbg_score(
            conn,
            run_id=run_id,
            composite_score=result.composite_score,
            grade=result.grade,
            breakdown_json=json.dumps(asdict(result), indent=2),
        )

        return result
    finally:
        conn.close()


def _grade_from_score(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"
