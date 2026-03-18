from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from irbg.analysis.reporting import build_run_report
from irbg.db.operations import DbConfig, connect, get_responses_for_run


class VisualizationError(Exception):
    """Raised when charts cannot be generated."""


def generate_run_summary_chart(
    *,
    db_path: Path,
    run_id: str,
    output_path: Path,
) -> None:
    report = build_run_report(db_path=db_path, run_id=run_id)

    if not report.pillar_scores:
        raise VisualizationError(
            f"Run '{run_id}' has no pillar scores to visualize."
        )

    labels = list(sorted(report.pillar_scores.keys()))
    values = [report.pillar_scores[label] for label in labels]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.ylim(0, 100)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Score")
    plt.title(
        f"IRBG Pillar Scores\nModel: {report.model_alias} | "
        f"Run: {report.run_id}"
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def generate_latency_chart(
    *,
    db_path: Path,
    run_id: str,
    output_path: Path,
) -> None:
    conn = connect(DbConfig(path=db_path))
    try:
        rows = get_responses_for_run(conn, run_id=run_id)
    finally:
        conn.close()

    if not rows:
        raise VisualizationError(
            f"Run '{run_id}' has no responses to visualize."
        )

    labels = [
        f"{row['scenario_id']}:{row['variant_id'] or '-'}" for row in rows
    ]
    values = [
        int(row["latency_ms"]) if row["latency_ms"] is not None else 0
        for row in rows
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(values)), values)
    plt.ylabel("Latency (ms)")
    plt.title(f"Response Latency by Prompt\nRun: {run_id}")
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
