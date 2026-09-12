# Rider Q-learning

A tiny tabular Q-learning agent that learns **hold vs release** for the game Rider. No neural network — just a 15×2 lookup table.

Works with **Ketchapp Rider** on Mac (`/Applications/Rider.app`).

## The idea (one paragraph)

Each frame we read the bike's **tilt angle** and **vertical velocity**, then **bin** those numbers into buckets (5 angle × 3 velocity = **15 states**). The Q-table stores how good `hold` and `release` have been from each state. The bot picks the better action (with some random exploration early on), gets a reward aligned with the real game (distance first, landed flips as a bonus), and nudges the table with the classic Q-learning update.

## How Rider scoring works (what we optimize)

- **Getting further** is the main score (sections / survival).
- Hold on the ground = accelerate; hold in the air = flip.
- A **landed** flip scores; a perfect landing roughly **doubles** flip points. Crashing mid-flip scores nothing and ends the run.
- So: go the distance; flip when you can land it, not for its own sake.

Our Mac reward mirrors that: `+1` per step alive, `+4` only after recovering upright from an inversion, `-12` on crash. Control steps are short (~40ms) so hold/release can be quick.

## Layout

```
train.py                 # train on the fake env (safe, no game needed)
train_mac.py             # train against real Rider on Mac
capture_test.py          # screenshot + angle smoke test
play_tap_test.py         # debug Play button click (saves annotated PNG)
rider_qlearning/
  bins.py                # continuous numbers → state id
  q_table.py             # the 15×2 table + update rule
  agent.py               # epsilon-greedy action choice
  mock_env.py            # fake Rider
  mac_window.py          # find window + screenshot
  mac_input.py           # mouse hold / release
  features.py            # angle + Play / in-run detection
  mac_env.py             # real-game environment
  safety.py              # kill switches
```

## Quick start (mock — no permissions)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

## Mac game setup

**Run Mac training from Terminal.app (or iTerm), not Cursor's terminal.**  
Screen Recording / Accessibility attach to the app that owns the process. Cursor's terminal counts as Cursor.

1. In Terminal.app:

```bash
cd ~/Desktop/rider-qlearning
source .venv/bin/activate
pip install -r requirements.txt
```

2. Open **Rider** and leave the game window visible.
3. Enable **Screen Recording** and **Accessibility** for **Terminal**, then quit/reopen Terminal.
4. Smoke-test:

```bash
python capture_test.py
python play_tap_test.py
```

5. Train:

```bash
python train_mac.py
```

Resumes from `checkpoints/q_table_mac.npy` when the reward version matches. A reward-function change auto-resets that checkpoint.

**Kill switches:** park mouse in top-left (~0.35s), hold Escape, or Ctrl+C.

## Bins

- angle edges `[-45, -15, 15, 45]` → 5 buckets  
- velocity edges `[-2, 2]` → 3 buckets  

State id = `angle_bin * 3 + velocity_bin` → ids `0..14`.
