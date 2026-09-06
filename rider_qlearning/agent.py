"""Epsilon-greedy policy over the Q-table."""

from __future__ import annotations

import random

from rider_qlearning.q_table import NUM_ACTIONS, QTable


class Agent:
    def __init__(self, q_table: QTable, epsilon: float = 1.0) -> None:
        self.q = q_table
        self.epsilon = epsilon  # probability of exploring at random

    def choose_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return random.randrange(NUM_ACTIONS)
        return self.q.best_action(state)

    def decay_epsilon(self, floor: float = 0.05, rate: float = 0.995) -> None:
        self.epsilon = max(floor, self.epsilon * rate)
