"""
Train on the real Rider Mac app (short runs — keep an eye on the mouse!).

Usage:
  1. Open Rider and get onto a screen where tapping starts / continues a run.
  2. Grant Screen Recording + Accessibility to your terminal.
  3. python train_mac.py

Move the mouse to a corner to abort if your OS has mouse-corner protection.
Press Ctrl+C to stop.
"""

from __future__ import annotations

import time
from pathlib import Path

from rider_qlearning.agent import Agent
from rider_qlearning.bins import discretize
from rider_qlearning.mac_env import MacRiderEnv
from rider_qlearning.q_table import ACTION_NAMES, QTable


EPISODES = 20
CHECKPOINT_DIR = Path("checkpoints")


def train() -> None:
    print("Starting in 3 seconds — switch to Rider…")
    time.sleep(3)

    env = MacRiderEnv()
    q_table = QTable(alpha=0.1, gamma=0.95)
    agent = Agent(q_table, epsilon=0.3)  # less random than mock; real flips hurt

    scores: list[float] = []

    for episode in range(1, EPISODES + 1):
        print(f"\n=== episode {episode}/{EPISODES} — tap to start a run if needed ===")
        time.sleep(1.5)
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
        agent.decay_epsilon(floor=0.05, rate=0.95)
        print(
            f"episode {episode:3d}  score={env.score:6.1f}  "
            f"steps={env.steps}  epsilon={agent.epsilon:.3f}"
        )

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    path = CHECKPOINT_DIR / "q_table_mac.npy"
    q_table.save(str(path))
    print(f"\nSaved Q-table to {path}")
    for s in range(q_table.q.shape[0]):
        best = q_table.best_action(s)
        print(f"  state {s:2d}: {ACTION_NAMES[best]:7s}  Q={q_table.q[s]}")


if __name__ == "__main__":
    train()
