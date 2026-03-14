from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModeOverlay:
    system_append: str = ""
    user_append: str = ""


@dataclass(frozen=True)
class ScenarioTemplate:
    id: str
    pillar: str
    category: str
    jurisdiction: str | None
    difficulty: str | None
    system_prompt_template: str
    user_prompt_template: str
    static_variables: dict[str, object] = field(default_factory=dict)
    variant_group: str | None = None
    modes: dict[str, ModeOverlay] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderedPrompt:
    scenario_id: str
    pillar: str
    category: str
    jurisdiction: str | None
    difficulty: str | None
    mode: str
    variant_id: str | None
    variables: dict[str, object]
    system_prompt: str
    user_prompt: str
