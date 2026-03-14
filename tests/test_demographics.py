from __future__ import annotations

from pathlib import Path

from irbg.demographics import get_variant_group, load_demographics_config


def test_load_demographics_config(tmp_path: Path) -> None:
    config_file = tmp_path / "demographics.yaml"
    config_file.write_text(
        """
variant_groups:
  test_group:
    - id: test_1
      name: Test User
      age: 34
      gender: male
      nationality: Indian
      religion: Hindu
      background: lives in Pune
      socioeconomic_signal: middle_class
"""
    )

    groups = load_demographics_config(path=config_file)

    assert "test_group" in groups
    assert len(groups["test_group"]) == 1
    assert groups["test_group"][0].name == "Test User"


def test_get_variant_group(tmp_path: Path) -> None:
    config_file = tmp_path / "demographics.yaml"
    config_file.write_text(
        """
variant_groups:
  demo_group:
    - id: demo_1
      name: Demo User
      age: 28
      gender: female
      nationality: Indian
      religion: Muslim
      background: lives in Mumbai
      socioeconomic_signal: middle_class
"""
    )

    variants = get_variant_group("demo_group", path=config_file)

    assert len(variants) == 1
    assert variants[0].id == "demo_1"
