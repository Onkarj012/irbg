from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from irbg.paths import CONFIG_DIR


class DemographicsError(Exception):
    """Raised when demographics configuration is missing or invalid."""


@dataclass(frozen=True)
class Variant:
    id: str
    group: str
    name: str
    age: int
    gender: str
    nationality: str
    religion: str
    background: str
    socioeconomic_signal: str | None = None

    def as_template_variables(self) -> dict[str, object]:
        variables: dict[str, object] = {
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "nationality": self.nationality,
            "religion": self.religion,
            "background": self.background,
        }

        if self.socioeconomic_signal is not None:
            variables["socioeconomic_signal"] = self.socioeconomic_signal

        return variables


def load_demographics_config(
    path: Path | None = None,
) -> dict[str, list[Variant]]:
    config_path = path or (CONFIG_DIR / "demographics.yaml")

    if not config_path.exists():
        raise DemographicsError(
            f"Demographics config file not found: {config_path}"
        )

    raw_data = yaml.safe_load(config_path.read_text()) or {}
    raw_groups = raw_data.get("variant_groups")

    if not isinstance(raw_groups, dict):
        raise DemographicsError(
            "Invalid demographics.yaml: expected 'variant_groups' mapping."
        )

    groups: dict[str, list[Variant]] = {}

    for group_name, items in raw_groups.items():
        if not isinstance(items, list):
            raise DemographicsError(
                f"Variant group '{group_name}' must be a list."
            )

        parsed_items: list[Variant] = []

        for item in items:
            if not isinstance(item, dict):
                raise DemographicsError(
                    f"Variant in group '{group_name}' must be a mapping."
                )

            try:
                parsed_items.append(
                    Variant(
                        id=str(item["id"]),
                        group=group_name,
                        name=str(item["name"]),
                        age=int(item["age"]),
                        gender=str(item["gender"]),
                        nationality=str(item["nationality"]),
                        religion=str(item["religion"]),
                        background=str(item["background"]),
                        socioeconomic_signal=_optional_str(
                            item.get("socioeconomic_signal")
                        ),
                    )
                )
            except KeyError as exc:
                raise DemographicsError(
                    f"Missing required key in group '{group_name}': {exc}"
                ) from exc

        groups[group_name] = parsed_items

    return groups


def get_variant_group(
    group_name: str,
    path: Path | None = None,
) -> list[Variant]:
    groups = load_demographics_config(path=path)

    try:
        return groups[group_name]
    except KeyError as exc:
        available = ", ".join(sorted(groups.keys()))
        raise DemographicsError(
            f"Unknown variant group '{group_name}'. Available: {available}"
        ) from exc


def get_variant_by_id(
    variant_id: str,
    path: Path | None = None,
) -> Variant:
    groups = load_demographics_config(path=path)

    for variants in groups.values():
        for variant in variants:
            if variant.id == variant_id:
                return variant

    raise DemographicsError(f"Unknown variant id '{variant_id}'.")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
