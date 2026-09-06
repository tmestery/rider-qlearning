# Rider Q-learning

A tiny tabular Q-learning agent that learns **hold vs release** for the game Rider. No neural network — just a 15×2 lookup table.

## The idea (one paragraph)

Each frame we read the bike's **tilt angle** and **vertical velocity**, then **bin** those numbers into buckets (5 angle × 3 velocity = **15 states**). The Q-table stores how good `hold` and `release` have been from each state. The bot picks the better action (with some random exploration early on), gets a **score-based reward** (flips that pay double are good), and nudges the table with the classic Q-learning update. After enough runs, the table becomes a cheat-sheet policy.

## Layout

```
train.py                 # run this to train
rider_qlearning/
  bins.py                # continuous numbers → state id
  q_table.py             # the 15×2 table + update rule
  agent.py               # epsilon-greedy action choice
  mock_env.py            # fake Rider (stand-in until real game wiring)
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

You should see average score print every 50 episodes, then a saved `checkpoints/q_table.npy`.

## Where we are

| Done | Not yet |
|------|---------|
| Binning (5×3) | Real screenshots → angle/velocity |
| Q-table + update | Real tap / hold injection |
| Epsilon-greedy agent | Playing against the actual Rider game |
| Mock env + train loop | Tuned bin edges for real physics |

Next step: decide how we talk to Rider (browser, desktop window, phone) and replace `mock_env.py` with that.

## Bins

Continuous values like `angle = 17.3` can't live in a small table. We chop the axis into ranges:

- angle edges `[-45, -15, 15, 45]` → 5 buckets  
- velocity edges `[-2, 2]` → 3 buckets  

State id = `angle_bin * 3 + velocity_bin` → ids `0..14`.
