"""
Real Rider environment on macOS (Ketchapp Rider.app).

Game scoring (research):
  - Distance / surviving sections is the main score.
  - Each landed flip is worth points; a perfect landing doubles flip points.
  - Airborne hold spins; grounded hold accelerates. Over-flipping crashes.

Our reward (aligned with that):
  - +1.0 each step alive  → getting further is the main signal
  - +4.0 only when a flip is *landed* (went inverted, then recovered upright)
  - -12.0 on crash       → reckless flips that kill the run are not worth it

Control is intentionally snappy (short step) so hold/release can flip fast.
"""

from __future__ import annotations

from rider_qlearning import features, mac_input, mac_window, safety
from rider_qlearning.q_table import HOLD
from rider_qlearning.step import StepResult

# Bump this when the reward meaning changes so trainers know to reset the table.
REWARD_VERSION = 2


class MacRiderEnv:
    def __init__(self, step_seconds: float = 0.04, max_steps: int = 600) -> None:
        self.step_seconds = step_seconds
        self.max_steps = max_steps
        self.window = mac_window.find_rider_window()
        self.holding = False
        self.prev_angle: float | None = None
        self.score = 0.0
        self.steps = 0
        self._flip_armed = False  # True after we were clearly inverted
        self._saw_bike = False

    def close(self) -> None:
        """Always release the mouse when stopping."""
        mac_input.emergency_release()
        self.holding = False

    def _tap_play(self, attempt: int = 0) -> None:
        """Tap Play (center lower-third) or a dismiss spot if Play isn't up yet."""
        mac_input.activate_rider()
        self.window = mac_window.find_rider_window()
        frame = mac_window.capture_window(self.window)

        found = features.find_play_button(frame)
        if found is not None:
            fx, fy = found
            source = "detected"
        else:
            # No Play yet (score / continue / ad) — tap center lower-third to advance.
            fx, fy = features.play_fallback_xy(frame)
            fy = int(frame.shape[0] * (0.72 + 0.04 * (attempt % 4)))
            source = "dismiss/fallback"

        x, y = self.window.frame_to_screen(fx, fy, frame.shape)
        print(
            f"  tap ({source}) frame=({fx},{fy}) screen=({x:.0f},{y:.0f})"
        )
        # Quick tap — no long AppleScript delay needed for every dismiss.
        mac_input.tap(x, y, hold_seconds=0.08, use_applescript=(attempt >= 3))

    def start_run(self) -> None:
        """
        Advance crash → continue → Play until an actual run is going.

        Important: the lobby bike must NOT count as a run (see features.in_run).
        """
        self.close()
        print("  tapping through to a new run…")
        for attempt in range(10):
            safety.check()
            self.window = mac_window.find_rider_window()
            frame = mac_window.capture_window(self.window)
            if features.in_run(frame):
                print(f"  run started (after {attempt} taps)")
                return

            self._tap_play(attempt)
            # Menu transitions need a beat; keep it short so training stays snappy.
            safety.sleep(0.85)

        print("  warning: still not in a run — continuing anyway")

    def reset(self) -> tuple[float, float]:
        """End any hold, restart a run, then read the starting state."""
        safety.check()
        self.close()
        safety.sleep(0.2)
        self.start_run()

        self.window = mac_window.find_rider_window()
        frame = mac_window.capture_window(self.window)
        angle = features.estimate_angle(frame)
        self.prev_angle = angle
        self.score = 0.0
        self.steps = 0
        self._flip_armed = False
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
        on_menu = features.on_menu(frame)
        angle = features.estimate_angle(frame)
        velocity = features.estimate_velocity(angle, self.prev_angle, self.step_seconds)
        prev = self.prev_angle
        self.prev_angle = angle
        self.steps += 1

        # Primary objective: get further (survive / cover ground).
        reward = 1.0
        done = False

        if visible:
            self._saw_bike = True

        # Arm flip bonus only after a clear inversion (airborne spin).
        if abs(angle) >= 120.0:
            self._flip_armed = True

        # Landed flip: were inverted, now mostly upright again → bonus (≈ double).
        # Reckless flips that never recover get no bonus and often crash below.
        if (
            self._flip_armed
            and prev is not None
            and abs(prev) >= 100.0
            and abs(angle) < 45.0
        ):
            reward += 4.0
            self._flip_armed = False

        # Crash / back to menu mid-run.
        if abs(angle) >= 170.0:
            reward = -12.0
            done = True
        elif self._saw_bike and (not visible or on_menu) and self.steps > 15:
            reward = -12.0
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
