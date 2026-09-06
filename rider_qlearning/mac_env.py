"""
Real Rider environment on macOS (Ketchapp Rider.app).

Each step:
  1. Hold or release the mouse over the game window.
  2. Wait a short beat.
  3. Screenshot → estimate angle / velocity.
  4. Reward from score-style signals (survive / flip / crash heuristic).

Before using this:
  - Open Rider and get to a playable run (or ready-to-tap screen).
  - Grant Screen Recording + Accessibility to your terminal.
"""

from __future__ import annotations

import time

from rider_qlearning import features, mac_input, mac_window
from rider_qlearning.q_table import HOLD
from rider_qlearning.step import StepResult


class MacRiderEnv:
    def __init__(self, step_seconds: float = 0.08, max_steps: int = 400) -> None:
        self.step_seconds = step_seconds
        self.max_steps = max_steps
        self.window = mac_window.find_rider_window()
        self.holding = False
        self.prev_angle: float | None = None
        self.score = 0.0
        self.steps = 0
        self._flip_armed = True  # only pay flip bonus once per rotation

    def reset(self) -> tuple[float, float]:
        """Release input and read the starting state. Start a run in-game first."""
        cx, cy = self.window.center
        if self.holding:
            mac_input.release(cx, cy)
            self.holding = False
        time.sleep(0.2)

        # Refresh window bounds in case it moved.
        self.window = mac_window.find_rider_window()
        frame = mac_window.capture_window(self.window)
        angle = features.estimate_angle(frame)
        self.prev_angle = angle
        self.score = 0.0
        self.steps = 0
        self._flip_armed = True
        return angle, 0.0

    def step(self, action: int) -> StepResult:
        cx, cy = self.window.center
        want_hold = action == HOLD

        if want_hold and not self.holding:
            mac_input.hold(cx, cy)
            self.holding = True
        elif not want_hold and self.holding:
            mac_input.release(cx, cy)
            self.holding = False
        # RELEASE action with already-released mouse: nothing to do.
        # HOLD with already-held mouse: keep holding.

        time.sleep(self.step_seconds)

        frame = mac_window.capture_window(self.window)
        angle = features.estimate_angle(frame)
        velocity = features.estimate_velocity(angle, self.prev_angle, self.step_seconds)
        self.prev_angle = angle
        self.steps += 1

        reward = 1.0  # survived another beat (score goes up while alive)
        done = False

        # Controlled flip: large |angle| then recovery → double points (your rule).
        if abs(angle) >= 140.0 and self._flip_armed:
            reward = 2.0
            self._flip_armed = False
        if abs(angle) < 60.0:
            self._flip_armed = True

        # Crash heuristic: extreme tilt that doesn't recover quickly.
        # (We'll refine this once we can detect the game-over screen.)
        if abs(angle) >= 170.0:
            reward = -5.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        if done and self.holding:
            mac_input.release(cx, cy)
            self.holding = False

        self.score += reward
        return StepResult(
            angle=angle,
            velocity=velocity,
            reward=reward,
            score=self.score,
            done=done,
        )
