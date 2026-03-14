from __future__ import annotations

import json
from pathlib import Path

from irbg.scenarios.template_models import ModeOverlay, ScenarioTemplate


class ScenarioTemplateLoadError(Exception):
    """Raised when a scenario template file is missing or invalid."""


def load_scenario_template(path: Path) -> ScenarioTemplate:
    if not path.exists():
        raise ScenarioTemplateLoadError(
            f"Scenario template file not found: {path}"
        )

    try:
        raw_data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ScenarioTemplateLoadError(
            f"Invalid JSON in scenario template file: {path}"
        ) from exc

    required_fields = [
        "id",
        "pillar",
        "category",
        "system_prompt_template",
        "user_prompt_template",
    ]

    missing_fields = [name for name in required_fields if name not in raw_data]

    if missing_fields:
        raise ScenarioTemplateLoadError(
            f"Scenario template file {path} is missing fields:"
            f"{','.join(missing_fields)}"
        )

    raw_modes = raw_data.get("modes", {})
    modes = _parse_modes(raw_modes)

    static_variables = raw_data.get("static_variables", {})
    if not isinstance(static_variables, dict):
        raise ScenarioTemplateLoadError(
            f"Invalid static_variables in scenario template file: {path}"
        )

    variant_group = raw_data.get("variant_group")
    if variant_group is not None:
        variant_group = str(variant_group)

    return ScenarioTemplate(
        id=str(raw_data["id"]),
        pillar=str(raw_data["pillar"]),
        category=str(raw_data["category"]),
        jurisdiction=_optional_str(raw_data.get("jurisdiction")),
        difficulty=_optional_str(raw_data.get("difficulty")),
        system_prompt_template=str(raw_data["system_prompt_template"]),
        user_prompt_template=str(raw_data["user_prompt_template"]),
        static_variables=static_variables,
        variant_group=variant_group,
        modes=modes,
    )


def _parse_modes(raw_modes: object) -> dict[str, ModeOverlay]:
    if not isinstance(raw_modes, dict):
        raise ScenarioTemplateLoadError(
            "Invalid modes value: Expected a mapping"
        )

    parsed: dict[str, ModeOverlay] = {}

    for mode_name, mode_value in raw_modes.items():
        if not isinstance(mode_value, dict):
            raise ScenarioTemplateLoadError(
                f"Invalid mode value for mode '{mode_name}': Expected a mapping"
            )

        parsed[str(mode_name)] = ModeOverlay(
            system_append=str(mode_value.get("system_append", "")),
            user_append=str(mode_value.get("user_append", "")),
        )

    return parsed


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
