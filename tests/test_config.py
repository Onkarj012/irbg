from __future__ import annotations

from pathlib import Path

from irbg.config import get_model_config, load_models_config


def test_load_models_config(tmp_path: Path) -> None:
    config_file = tmp_path / "models.yaml"
    config_file.write_text(
        """
models:
  test-model:
    name: Test Model
    provider: openrouter
    model_id: provider/test-model
    max_tokens: 256
    temperature: 0.0
"""
    )

    models = load_models_config(path=config_file)

    assert "test-model" in models
    assert models["test-model"].name == "Test Model"
    assert models["test-model"].model_id == "provider/test-model"


def test_get_model_config(tmp_path: Path) -> None:
    config_file = tmp_path / "models.yaml"
    config_file.write_text(
        """
models:
  alpha:
    name: Alpha Model
    provider: openrouter
    model_id: provider/alpha
    max_tokens: 512
    temperature: 0.1
"""
    )

    model = get_model_config("alpha", path=config_file)

    assert model.alias == "alpha"
    assert model.max_tokens == 512
    assert model.temperature == 0.1
