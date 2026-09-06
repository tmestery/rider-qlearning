"""Turn continuous bike readings into a small discrete state id."""

from __future__ import annotations

# Angle bins (degrees). Negative = leaning back, positive = leaning forward.
# Edges split the number line into 5 buckets.
ANGLE_EDGES = [-45.0, -15.0, 15.0, 45.0]

# Vertical velocity bins. Edges split into 3 buckets.
VELOCITY_EDGES = [-2.0, 2.0]

NUM_ANGLE_BINS = len(ANGLE_EDGES) + 1  # 5
NUM_VELOCITY_BINS = len(VELOCITY_EDGES) + 1  # 3
NUM_STATES = NUM_ANGLE_BINS * NUM_VELOCITY_BINS  # 15


def _bin_index(value: float, edges: list[float]) -> int:
    """Return which bucket `value` falls into given sorted edge list."""
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges)


def discretize(angle: float, velocity: float) -> int:
    """Map (angle, velocity) -> state id in [0, NUM_STATES)."""
    a = _bin_index(angle, ANGLE_EDGES)
    v = _bin_index(velocity, VELOCITY_EDGES)
    return a * NUM_VELOCITY_BINS + v
