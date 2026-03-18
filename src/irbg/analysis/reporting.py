from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from irbg.analysis.aggregate import aggregate_run_score
from irbg.db.operations import (
    DbConfig,
    connect,
    get_all_pillar_scores,
    get_irbg_score,
    get_responses_for_run,
    get_run,
)


@dataclass(frozen=True)
class RunReport:
    run_id: str
    model_alias: str
    mode: str
    status: str
    response_count: int
    scenario_count: int
    average_latency_ms: float
    average_tokens: float
    pillar_scores: dict[str, float]
    composite_score: float | None
    grade: str | None


class RunReportError(Exception):
    """Raised when a run report cannot be generated."""


def build_run_report(
    *,
    db_path: Path,
    run_id: str,
) -> RunReport:
    conn = connect(DbConfig(path=db_path))

    try:
        run_row = get_run(conn, run_id=run_id)
        if run_row is None:
            raise RunReportError(f"Run not found: {run_id}")

        response_rows = get_responses_for_run(conn, run_id=run_id)
        pillar_rows = get_all_pillar_scores(conn, run_id=run_id)
        irbg_row = get_irbg_score(conn, run_id=run_id)

        response_count = len(response_rows)
        scenario_count = len({row["scenario_id"] for row in response_rows})

        latencies = [
            int(row["latency_ms"])
            for row in response_rows
            if row["latency_ms"] is not None
        ]
        tokens = [
            int(row["response_tokens"])
            for row in response_rows
            if row["response_tokens"] is not None
        ]

        average_latency_ms = (
            round(sum(latencies) / len(latencies), 2) if latencies else 0.0
        )
        average_tokens = round(sum(tokens) / len(tokens), 2) if tokens else 0.0

        pillar_scores = {
            str(row["pillar"]): float(row["score"]) for row in pillar_rows
        }

        if irbg_row is None and pillar_scores:
            aggregate_run_score(db_path=db_path, run_id=run_id)
            irbg_row = get_irbg_score(conn, run_id=run_id)

        return RunReport(
            run_id=run_id,
            model_alias=str(run_row["model_id"]),
            mode=str(run_row["mode"]),
            status=str(run_row["status"]),
            response_count=response_count,
            scenario_count=scenario_count,
            average_latency_ms=average_latency_ms,
            average_tokens=average_tokens,
            pillar_scores=pillar_scores,
            composite_score=float(irbg_row["composite_score"])
            if irbg_row is not None
            else None,
            grade=str(irbg_row["grade"]) if irbg_row is not None else None,
        )
    finally:
        conn.close()


def write_run_report_json(
    *,
    report: RunReport,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(report), indent=2))


def write_run_report_markdown(
    *,
    report: RunReport,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        f"# IRBG Run Report — {report.run_id}",
        "",
        f"- Model: `{report.model_alias}`",
        f"- Mode: `{report.mode}`",
        f"- Status: `{report.status}`",
        f"- Response Count: `{report.response_count}`",
        f"- Scenario Count: `{report.scenario_count}`",
        f"- Average Latency (ms): `{report.average_latency_ms}`",
        f"- Average Tokens: `{report.average_tokens}`",
        f"- Composite Score: `{report.composite_score}`",
        f"- Grade: `{report.grade}`",
        "",
        "## Pillar Scores",
        "",
    ]

    if report.pillar_scores:
        for pillar, score in sorted(report.pillar_scores.items()):
            lines.append(f"- `{pillar}`: `{score}`")
    else:
        lines.append("- No pillar scores available")

    lines.append("")

    output_path.write_text("\n".join(lines))
