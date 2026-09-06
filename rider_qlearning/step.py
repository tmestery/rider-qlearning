"""Shared step result used by mock and Mac environments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepResult:
    angle: float
    velocity: float
    reward: float
    score: float
    done: bool
