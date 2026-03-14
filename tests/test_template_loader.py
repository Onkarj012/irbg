from __future__ import annotations

from pathlib import Path

from irbg.scenarios.template_loader import load_scenario_template


def test_load_scenario_template(tmp_path: Path) -> None:
    template_file = tmp_path / "template.json"
    template_file.write_text(
        """
        {
        "id": "P1-TEST-001",
        "pillar": "p1_demographic_consistency",
        "category": "bail_recommendation",
        "jurisdiction": "US",
        "difficulty": "medium",
        "system_prompt_template": "You are a legal advisor.",
        "user_prompt_template": "Review {name}.",
        "static_variables": {
            "case_num": "123"
        },
        "variant_group": "us_core",
        "modes": {
            "pressure": {
            "system_append": "Be fast.",
            "user_append": "Respond now."
            }
        }
        }
        """
    )

    template = load_scenario_template(template_file)

    assert template.id == "P1-TEST-001"
    assert template.variant_group == "us_core"
