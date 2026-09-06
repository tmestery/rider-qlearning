"""
Train on the real Rider Mac app (short runs).

Kill switches (any one stops training and releases the mouse):
  - Flick the cursor into the TOP-LEFT corner of the screen
  - Hold Escape
  - Ctrl+C

Usage:
  1. Open Rider and get onto a screen where tapping starts / continues a run.
  2. Grant Screen Recording + Accessibility to your terminal.
  3. python train_mac.py
"""

from __future__ import annotations

from pathlib import Path

from rider_qlearning.agent import Agent
from rider_qlearning.bins import discretize
from rider_qlearning.mac_env import MacRiderEnv
from rider_qlearning.q_table import ACTION_NAMES, QTable
from rider_qlearning.safety import AbortTraining, arm, shutdown, sleep


EPISODES = 20
CHECKPOINT_DIR = Path("checkpoints")


def train() -> None:
    print(
        "\n"
        "=== SAFETY ===\n"
        "  Take control back anytime:\n"
        "    • park mouse in TOP-LEFT corner (~0.35s)\n"
        "    • hold Escape\n"
        "    • Ctrl+C\n"
        "  Any of these releases the mouse and stops training.\n"
        "  Episodes auto-tap Play / Play again between runs.\n"
        "==============\n"
    )
    print("Starting in 3 seconds — switch to Rider…")
    arm()
    sleep(3)

    env = MacRiderEnv()
    q_table = QTable(alpha=0.1, gamma=0.95)
    agent = Agent(q_table, epsilon=0.3)

    try:
        for episode in range(1, EPISODES + 1):
            print(f"\n=== episode {episode}/{EPISODES} ===")
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

    except AbortTraining as exc:
        print(f"Training aborted ({exc}). You have control again.")
    finally:
        env.close()
        shutdown()


if __name__ == "__main__":
    train()
