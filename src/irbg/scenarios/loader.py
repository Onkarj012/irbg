from __future__ import annotations

import json
from pathlib import Path

from irbg.scenarios.models import Scenario


class ScenarioLoadError(Exception):
    """Raised when a scenario file is missing or invalid."""


def load_scenario(path: Path) -> Scenario:
    if not path.exists():
        raise ScenarioLoadError(f"Scenario file not found: {path}")

    try:
        raw_data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ScenarioLoadError(
            f"Invalid JSON in scenario file: {path}"
        ) from exc

    required_fields = [
        "id",
        "pillar",
        "category",
        "system_prompt",
        "user_prompt",
    ]

    missing_fields = [
        field for field in required_fields if field not in raw_data
    ]
    if missing_fields:
        raise ScenarioLoadError(
            f"Scenario file {path} is missing fields: "
            f"{', '.join(missing_fields)}"
        )

    return Scenario(
        id=str(raw_data["id"]),
        pillar=str(raw_data["pillar"]),
        category=str(raw_data["category"]),
        jurisdiction=_optional_str(raw_data.get("jurisdiction")),
        difficulty=_optional_str(raw_data.get("difficulty")),
        system_prompt=str(raw_data["system_prompt"]),
        user_prompt=str(raw_data["user_prompt"]),
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
