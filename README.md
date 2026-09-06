# Rider Q-learning

A tiny tabular Q-learning agent that learns **hold vs release** for the game Rider. No neural network — just a 15×2 lookup table.

Works with **Ketchapp Rider** on Mac (`/Applications/Rider.app`).

## The idea (one paragraph)

Each frame we read the bike's **tilt angle** and **vertical velocity**, then **bin** those numbers into buckets (5 angle × 3 velocity = **15 states**). The Q-table stores how good `hold` and `release` have been from each state. The bot picks the better action (with some random exploration early on), gets a **score-based reward** (flips that pay double are good), and nudges the table with the classic Q-learning update. After enough runs, the table becomes a cheat-sheet policy.

## Layout

```
train.py                 # train on the fake env (safe, no game needed)
train_mac.py             # train against real Rider on Mac
capture_test.py          # screenshot + angle smoke test
rider_qlearning/
  bins.py                # continuous numbers → state id
  q_table.py             # the 15×2 table + update rule
  agent.py               # epsilon-greedy action choice
  mock_env.py            # fake Rider
  mac_window.py          # find window + screenshot
  mac_input.py           # mouse hold / release
  features.py            # angle from silhouette (PCA)
  mac_env.py             # real-game environment
```

## Quick start (mock — no permissions)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

## Mac game setup

1. Install deps (above).
2. Open **Rider** and leave the game window visible.
3. Grant your terminal (or Cursor) **Screen Recording** and **Accessibility**:
   System Settings → Privacy & Security.
4. Smoke-test capture:

```bash
python capture_test.py
```

You should get `captures/rider_frame.png` and a printed angle.

5. Train on the real game:

```bash
python train_mac.py
```

**Kill switches** (any one stops training and releases the mouse):

- Move the cursor to the **top-left corner** of the screen (hold ~0.35s)
- Hold **Escape**
- **Ctrl+C**

Between episodes the bot **taps Play / Play again** for you (often twice — score screen then restart).

## Where we are

| Done | Still rough |
|------|-------------|
| Mock train loop | Angle estimate is a first-pass silhouette PCA |
| Mac screenshot + hold/release | Crash / flip detection is heuristic, not OCR score |
| Binning + Q-learning | Restarting runs between episodes is still manual |
| `capture_test.py` / `train_mac.py` | Bin edges may need tuning for real physics |

## Bins

Continuous values like `angle = 17.3` can't live in a small table. We chop the axis into ranges:

- angle edges `[-45, -15, 15, 45]` → 5 buckets  
- velocity edges `[-2, 2]` → 3 buckets  

State id = `angle_bin * 3 + velocity_bin` → ids `0..14`.
