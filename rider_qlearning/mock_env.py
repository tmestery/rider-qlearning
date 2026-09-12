"""
Fake Rider environment for learning the training loop.

Hold leans forward / flips in air; release levels out.
Distance (steps alive) is the main reward; a landed flip pays a bonus;
crashing is heavily penalized (same idea as the Mac env).
"""

from __future__ import annotations

import random

from rider_qlearning.step import StepResult


class MockRiderEnv:
    def __init__(self) -> None:
        self.angle = 0.0
        self.velocity = 0.0
        self.score = 0.0
        self.steps = 0
        self.max_steps = 200
        self._flip_armed = False

    def reset(self) -> tuple[float, float]:
        self.angle = random.uniform(-10.0, 10.0)
        self.velocity = 0.0
        self.score = 0.0
        self.steps = 0
        self._flip_armed = False
        return self.angle, self.velocity

    def step(self, action: int) -> StepResult:
        accel = 3.0 if action == 0 else -3.0
        self.velocity += accel + random.uniform(-0.5, 0.5)
        self.velocity *= 0.9
        prev = self.angle
        self.angle += self.velocity
        self.steps += 1

        reward = 1.0  # get further
        done = False

        if abs(self.angle) >= 120.0:
            self._flip_armed = True

        # Landed flip bonus (recovered upright after inversion).
        if self._flip_armed and abs(prev) >= 100.0 and abs(self.angle) < 45.0:
            reward += 4.0
            self._flip_armed = False
            self.angle = (self.angle % 360.0) - 180.0

        if abs(self.angle) >= 200.0:
            reward = -12.0
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
