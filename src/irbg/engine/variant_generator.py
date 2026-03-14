from __future__ import annotations

from pathlib import Path

from irbg.demographics import (
    get_variant_by_id,
    get_variant_group,
)
from irbg.engine.prompt_builder import render_prompt
from irbg.scenarios.template_models import (
    RenderedPrompt,
    ScenarioTemplate,
)


class VariantGenerationError(Exception):
    """Raised when demographic variants cannot be generated."""


def generate_prompts_for_template(
    template: ScenarioTemplate,
    *,
    mode: str = "baseline",
    demographics_path: Path | None = None,
) -> list[RenderedPrompt]:
    if not template.variant_group:
        raise VariantGenerationError(
            f"Template '{template.id}' has no variant_group defined."
        )

    variants = get_variant_group(
        template.variant_group,
        path=demographics_path,
    )

    return [
        render_prompt(
            template,
            variables=variant.as_template_variables(),
            mode=mode,
            variant_id=variant.id,
        )
        for variant in variants
    ]


def generate_single_prompt_for_variant(
    template: ScenarioTemplate,
    *,
    variant_id: str,
    mode: str = "baseline",
    demographics_path: Path | None = None,
) -> RenderedPrompt:
    if not template.variant_group:
        raise VariantGenerationError(
            f"Template '{template.id}' has no variant_group defined."
        )

    variant = get_variant_by_id(
        variant_id,
        path=demographics_path,
    )

    if variant.group != template.variant_group:
        raise VariantGenerationError(
            f"Variant '{variant.id}' belongs to group '{variant.group}', "
            f"but template '{template.id}' expects group "
            f"'{template.variant_group}'."
        )

    return render_prompt(
        template,
        variables=variant.as_template_variables(),
        mode=mode,
        variant_id=variant.id,
    )
