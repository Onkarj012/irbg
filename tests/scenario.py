from __future__ import annotations

from pathlib import Path

from irbg.scenarios.loader import load_scenario


def test_load_scenario(tmp_path: Path) -> None:
    scenario_file = tmp_path / "scenario.json"
    scenario_file.write_text(
        """
{
  "id": "TEST-001",
  "pillar": "smoke",
  "category": "demo",
  "jurisdiction": "general",
  "difficulty": "easy",
  "system_prompt": "You are helpful.",
  "user_prompt": "Say hello."
}
"""
    )

    scenario = load_scenario(scenario_file)

    assert scenario.id == "TEST-001"
    assert scenario.pillar == "smoke"
    assert scenario.category == "demo"
    assert scenario.system_prompt == "You are helpful."
    assert scenario.user_prompt == "Say hello."
