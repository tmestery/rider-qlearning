# Changelog

## Unreleased

## v2.0.0

- Reward v2: distance-first (`+1`/step), landed-flip bonus (`+4`), harsh crash (`-12`); snappier ~40ms control.
- Fix Play-again: require real in-run (Play button gone), reject off-center green, dismiss score screens.
- Resume checkpoints; auto-reset table when reward version changes.
- Run Mac training from Terminal.app (not Cursor) so only Terminal needs Screen Recording.
- Kill switches: top-left corner, Escape, Ctrl+C; emergency mouse release.
- Public-ready docs: README with Rider screenshot, architecture diagram, `pyproject.toml`.

## v1

- Mac Rider wiring: window capture, mouse hold/release, silhouette angle estimate, `MacRiderEnv`, `capture_test.py`, `train_mac.py`.
- Initial scaffold: bins, Q-table, epsilon-greedy agent, mock Rider env, train loop.
