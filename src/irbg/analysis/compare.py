from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from irbg.analysis.reporting import build_run_report


@dataclass(frozen=True)
class RunComparison:
    left_run_id: str
    right_run_id: str
    left_model: str
    right_model: str
    left_score: float | None
    right_score: float | None
    score_delta: float | None
    left_grade: str | None
    right_grade: str | None


def compare_runs(
    *,
    db_path: Path,
    left_run_id: str,
    right_run_id: str,
) -> RunComparison:
    left = build_run_report(db_path=db_path, run_id=left_run_id)
    right = build_run_report(db_path=db_path, run_id=right_run_id)

    if left.composite_score is None or right.composite_score is None:
        delta = None
    else:
        delta = round(left.composite_score - right.composite_score, 2)

    return RunComparison(
        left_run_id=left.run_id,
        right_run_id=right.run_id,
        left_model=left.model_alias,
        right_model=right.model_alias,
        left_score=left.composite_score,
        right_score=right.composite_score,
        score_delta=delta,
        left_grade=left.grade,
        right_grade=right.grade,
    )
