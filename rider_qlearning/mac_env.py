"""
Real Rider environment on macOS (Ketchapp Rider.app).

Each step:
  1. Hold or release the mouse over the game window.
  2. Wait a short beat (interruptible — kill switches checked).
  3. Screenshot → estimate angle / velocity.
  4. Reward from score-style signals (survive / flip / crash heuristic).

When a run ends, reset() taps the window to hit Play / Play again.
"""

from __future__ import annotations

from rider_qlearning import features, mac_input, mac_window, safety
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
        self._flip_armed = True
        self._saw_bike = False  # becomes True once silhouette appears this episode

    def close(self) -> None:
        """Always release the mouse when stopping."""
        mac_input.emergency_release()
        self.holding = False

    def _tap_play(self, attempt: int = 0) -> None:
        """Find Play in the lower third (or use a fixed spot) and tap it."""
        mac_input.activate_rider()
        self.window = mac_window.find_rider_window()
        frame = mac_window.capture_window(self.window)

        found = features.find_play_button(frame)
        if found is None:
            # Fixed lower-third middle; nudge a bit on retries.
            fx, fy = features.play_fallback_xy(frame)
            fy = int(frame.shape[0] * (0.76 + 0.04 * (attempt % 3)))
            source = "fallback"
        else:
            fx, fy = found
            source = "detected"

        x, y = self.window.frame_to_screen(fx, fy, frame.shape)
        print(
            f"  tap Play ({source}) frame=({fx},{fy}) screen=({x:.0f},{y:.0f}) "
            f"window=({self.window.left:.0f},{self.window.top:.0f},"
            f"{self.window.width:.0f}x{self.window.height:.0f})"
        )
        mac_input.tap(x, y, hold_seconds=0.15)

    def start_run(self) -> None:
        """
        Tap through Play / Play again until a bike silhouette shows up.
        Rider often needs: crash → tap (score) → tap (play again).
        """
        self.close()
        print("  tapping Play / Play again (lower third)…")
        for attempt in range(6):
            safety.check()
            self._tap_play(attempt)
            safety.sleep(1.2)
            self.window = mac_window.find_rider_window()
            frame = mac_window.capture_window(self.window)
            if features.bike_visible(frame):
                print(f"  run started (tap {attempt + 1})")
                return
        print("  warning: no bike seen yet — continuing anyway")

    def reset(self) -> tuple[float, float]:
        """End any hold, tap to (re)start a run, then read the starting state."""
        safety.check()
        self.close()
        safety.sleep(0.3)
        self.start_run()

        self.window = mac_window.find_rider_window()
        frame = mac_window.capture_window(self.window)
        angle = features.estimate_angle(frame)
        self.prev_angle = angle
        self.score = 0.0
        self.steps = 0
        self._flip_armed = True
        self._saw_bike = features.bike_visible(frame)
        return angle, 0.0

    def step(self, action: int) -> StepResult:
        safety.check()
        cx, cy = self.window.center
        want_hold = action == HOLD

        if want_hold and not self.holding:
            mac_input.hold(cx, cy)
            self.holding = True
        elif not want_hold and self.holding:
            mac_input.release(cx, cy)
            self.holding = False

        safety.sleep(self.step_seconds)

        frame = mac_window.capture_window(self.window)
        visible = features.bike_visible(frame)
        angle = features.estimate_angle(frame)
        velocity = features.estimate_velocity(angle, self.prev_angle, self.step_seconds)
        self.prev_angle = angle
        self.steps += 1

        reward = 1.0
        done = False

        if visible:
            self._saw_bike = True

        # Controlled flip: large |angle| then recovery → double points.
        if abs(angle) >= 140.0 and self._flip_armed:
            reward = 2.0
            self._flip_armed = False
        if abs(angle) < 60.0:
            self._flip_armed = True

        # Crash: extreme tilt, or bike disappeared after we had seen it (game over UI).
        if abs(angle) >= 170.0:
            reward = -5.0
            done = True
        elif self._saw_bike and not visible and self.steps > 10:
            reward = -5.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        if done:
            self.close()

        self.score += reward
        return StepResult(
            angle=angle,
            velocity=velocity,
            reward=reward,
            score=self.score,
            done=done,
        )
