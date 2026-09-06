"""Tiny Q-table: one row per state, two columns (hold / release)."""

from __future__ import annotations

import numpy as np

from rider_qlearning.bins import NUM_STATES

# Actions
HOLD = 0
RELEASE = 1
NUM_ACTIONS = 2
ACTION_NAMES = ("hold", "release")


class QTable:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.95) -> None:
        self.alpha = alpha  # learning rate: how hard each update nudges the table
        self.gamma = gamma  # discount: how much we care about future rewards
        self.q = np.zeros((NUM_STATES, NUM_ACTIONS), dtype=np.float64)

    def best_action(self, state: int) -> int:
        return int(np.argmax(self.q[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """Classic Q-learning: Q(s,a) += alpha * (target - Q(s,a))."""
        old = self.q[state, action]
        future = 0.0 if done else float(np.max(self.q[next_state]))
        target = reward + self.gamma * future
        self.q[state, action] = old + self.alpha * (target - old)

    def save(self, path: str) -> None:
        np.save(path, self.q)

    def load(self, path: str) -> None:
        self.q = np.load(path)
