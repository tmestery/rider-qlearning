"""
Fake Rider environment for learning the training loop.

Later we replace this with real screenshots + tap injection.
For now: hold leans the bike forward, release leans it back.
Score goes up each step; a controlled flip pays double; a crash ends the run.
"""

from __future__ import annotations

from dataclasses import dataclass

import random


@dataclass
class StepResult:
    angle: float
    velocity: float
    reward: float
    score: float
    done: bool


class MockRiderEnv:
    def __init__(self) -> None:
        self.angle = 0.0
        self.velocity = 0.0
        self.score = 0.0
        self.steps = 0
        self.max_steps = 200

    def reset(self) -> tuple[float, float]:
        self.angle = random.uniform(-10.0, 10.0)
        self.velocity = 0.0
        self.score = 0.0
        self.steps = 0
        return self.angle, self.velocity

    def step(self, action: int) -> StepResult:
        # action 0 = hold (lean forward), 1 = release (lean back)
        accel = 3.0 if action == 0 else -3.0
        self.velocity += accel + random.uniform(-0.5, 0.5)
        self.velocity *= 0.9  # a little damping
        self.angle += self.velocity
        self.steps += 1

        reward = 1.0  # base points for surviving a step
        done = False

        # Controlled flip near a full rotation: double points, keep going.
        if abs(self.angle) >= 160.0 and abs(self.angle) < 200.0:
            reward = 2.0
            self.angle = (self.angle % 360.0) - 180.0  # wrap after flip

        # Crash: over-rotated past recovery.
        if abs(self.angle) >= 200.0:
            reward = -5.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        self.score += reward
        return StepResult(
            angle=self.angle,
            velocity=self.velocity,
            reward=reward,
            score=self.score,
            done=done,
        )
