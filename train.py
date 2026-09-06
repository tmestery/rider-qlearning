"""
Train a tiny Q-learning agent on the mock Rider environment.

Run from the repo root:
    python train.py
"""

from __future__ import annotations

from pathlib import Path

from rider_qlearning.agent import Agent
from rider_qlearning.bins import discretize
from rider_qlearning.mock_env import MockRiderEnv
from rider_qlearning.q_table import ACTION_NAMES, QTable


EPISODES = 500
CHECKPOINT_DIR = Path("checkpoints")


def train() -> None:
    env = MockRiderEnv()
    q_table = QTable(alpha=0.1, gamma=0.95)
    agent = Agent(q_table, epsilon=1.0)

    scores: list[float] = []

    for episode in range(1, EPISODES + 1):
        angle, velocity = env.reset()
        state = discretize(angle, velocity)
        done = False

        while not done:
            action = agent.choose_action(state)
            result = env.step(action)
            next_state = discretize(result.angle, result.velocity)

            q_table.update(
                state=state,
                action=action,
                reward=result.reward,
                next_state=next_state,
                done=result.done,
            )

            state = next_state
            done = result.done

        scores.append(env.score)
        agent.decay_epsilon()

        if episode % 50 == 0:
            avg = sum(scores[-50:]) / 50
            print(
                f"episode {episode:4d}  "
                f"avg_score={avg:6.1f}  "
                f"epsilon={agent.epsilon:.3f}"
            )

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    path = CHECKPOINT_DIR / "q_table.npy"
    q_table.save(str(path))
    print(f"\nSaved Q-table to {path}")
    print("Learned preference per state (best action):")
    for s in range(q_table.q.shape[0]):
        best = q_table.best_action(s)
        print(f"  state {s:2d}: {ACTION_NAMES[best]:7s}  Q={q_table.q[s]}")


if __name__ == "__main__":
    train()
