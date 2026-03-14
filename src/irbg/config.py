from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from irbg.paths import CONFIG_DIR


class ConfigError(Exception):
    """Raised when configuration files are missing or invalid."""


@dataclass
class ModelConfig:
    alias: str
    name: str
    provider: str
    model_id: str
    max_tokens: int
    temperature: float


def load_model_config(
    path: Path | None = None,
) -> dict[str, ModelConfig]:
    config_path = path or (CONFIG_DIR / "models.yaml")

    if not config_path.exists():
        raise ConfigError(f"Model config file not found at {config_path}")

    raw_data = yaml.safe_load(config_path.read_text()) or {}
    raw_models = raw_data.get("models")

    if not isinstance(raw_models, dict):
        raise ConfigError(
            "Invalid models.yaml: expected a top-level 'models' mapping."
        )

    models: dict[str, ModelConfig] = {}

    for alias, values in raw_models.items():
        if not isinstance(values, dict):
            raise ConfigError(
                "Invalid models.yaml: expected a mapping for model alias "
                f"'{alias}'"
            )

        try:
            models[alias] = ModelConfig(
                alias=alias,
                name=str(values["name"]),
                provider=str(values["provider"]),
                model_id=str(values["model_id"]),
                max_tokens=int(values["max_tokens"]),
                temperature=float(values["temperature"]),
            )
        except KeyError as exc:
            raise ConfigError(
                f"Missing required Key for model '{alias}': {exc}"
            ) from exc

    return models


def load_models_config(
    path: Path | None = None,
) -> dict[str, ModelConfig]:
    """
    Backwards-compatible alias for loading all model configurations.

    Kept for compatibility with earlier APIs and existing tests that
    expect a pluralized function name.
    """
    return load_model_config(path=path)


def get_model_config(
    alias: str,
    path: Path | None = None,
) -> ModelConfig:
    models = load_model_config(path=path)

    try:
        return models[alias]
    except KeyError as exc:
        available = ", ".join(sorted(models.keys()))
        raise ConfigError(
            f"Unknown model alias '{alias}'. Available models: {available}"
        ) from exc
