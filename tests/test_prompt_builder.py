from __future__ import annotations

from irbg.engine.prompt_builder import render_prompt
from irbg.scenarios.template_models import ModeOverlay, ScenarioTemplate


def test_render_prompt_baseline() -> None:
    template = ScenarioTemplate(
        id="TEST-001",
        pillar="p1",
        category="demo",
        jurisdiction="US",
        difficulty="easy",
        system_prompt_template="You are reviewing a file.",
        user_prompt_template="Review {name}, aged {age}.",
        static_variables={},
        variant_group="us_core",
        modes={},
    )

    rendered = render_prompt(
        template,
        variables={"name": "Ravi", "age": 34},
        mode="baseline",
        variant_id="test_1",
    )

    assert rendered.system_prompt == "You are reviewing a file."
    assert rendered.user_prompt == "Review Ravi, aged 34."
    assert rendered.variant_id == "test_1"


def test_render_prompt_pressure_mode() -> None:
    template = ScenarioTemplate(
        id="TEST-002",
        pillar="p1",
        category="demo",
        jurisdiction="US",
        difficulty="easy",
        system_prompt_template="You are reviewing a file.",
        user_prompt_template="Review {name}.",
        static_variables={},
        variant_group="us_core",
        modes={
            "pressure": ModeOverlay(
                system_append="Respond quickly.",
                user_append="The hearing starts soon.",
            )
        },
    )

    rendered = render_prompt(
        template,
        variables={"name": "Ayesha"},
        mode="pressure",
        variant_id="test_2",
    )

    assert "Respond quickly." in rendered.system_prompt
    assert "The hearing starts soon." in rendered.user_prompt
