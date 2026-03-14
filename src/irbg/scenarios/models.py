from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Scenario:
    id: str
    pillar: str
    category: str
    jurisdiction: str
    difficulty: str
    system_prompt: str
    user_prompt: str
