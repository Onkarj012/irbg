from __future__ import annotations

from irbg.scenarios.template_models import (
    ModeOverlay,
    RenderedPrompt,
    ScenarioTemplate,
)


class PromptBuildError(Exception):
    """Raised when a prompt cannot be rendered from a template."""


def render_prompt(
    template: ScenarioTemplate,
    *,
    variables: dict[str, object],
    mode: str,
    variant_id: str | None = None,
) -> RenderedPrompt:
    merged_variables = {**template.static_variables, **variables}

    overlay = _resolve_mode_overlay(template, mode)

    try:
        system_prompt = template.system_prompt_template.format(
            **merged_variables
        )

        user_prompt = template.user_prompt_template.format(**merged_variables)
    except KeyError as exc:
        raise PromptBuildError(
            f"Missing variable '{exc.args[0]}' while rendering "
            f"template '{template.id}'."
        ) from exc

    if overlay.system_append:
        system_prompt = f"{system_prompt}\n\n{overlay.system_append}"

    if overlay.user_append:
        user_prompt = f"{user_prompt}\n\n{overlay.user_append}"

    return RenderedPrompt(
        scenario_id=template.id,
        pillar=template.pillar,
        category=template.category,
        jurisdiction=template.jurisdiction,
        difficulty=template.difficulty,
        mode=mode,
        variant_id=variant_id,
        variables=merged_variables,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _resolve_mode_overlay(
    template: ScenarioTemplate,
    mode: str,
) -> ModeOverlay:
    if mode == "baseline":
        return ModeOverlay()

    try:
        return template.modes[mode]
    except KeyError as exc:
        available = ", ".join(["baseline", *sorted(template.modes.keys())])
        raise PromptBuildError(
            f"Mode '{mode}' is not defined for template '{template.id}'. "
            f"Available modes: {available}"
        ) from exc
