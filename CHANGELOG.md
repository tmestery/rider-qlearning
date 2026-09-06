# Changelog

## v1

- Mac Rider wiring: window capture, mouse hold/release, silhouette angle estimate, `MacRiderEnv`, `capture_test.py`, `train_mac.py`.
- Initial scaffold: bins, Q-table, epsilon-greedy agent, mock Rider env, train loop.

## Unreleased

- Auto tap Play / Play again between Mac episodes; detect run end when bike silhouette vanishes.
- Debounce top-left kill switch (~0.35s) to avoid accidental aborts.
- Kill switches for Mac training: top-left corner, Escape, Ctrl+C; always release mouse on exit.
