"""
Estimate bike tilt from a Rider screenshot.

Approach (simple on purpose):
1. Crop the middle of the frame (where the bike usually is).
2. Find dark pixels (the rider silhouette).
3. Fit a line through those pixels with PCA — that line's angle is our tilt.

Velocity is just how fast that angle is changing between frames.
"""

from __future__ import annotations

import numpy as np


def estimate_angle(frame: np.ndarray) -> float:
    """
    Return tilt in degrees. 0 ≈ upright, positive ≈ leaning forward (nose down).
    Returns 0.0 if we can't find the bike.
    """
    xs, ys = _silhouette_points(frame)
    if len(xs) < 50:
        return 0.0

    # PCA: first principal component ≈ bike long axis.
    pts = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    pts -= pts.mean(axis=0)
    _cov = np.cov(pts, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(_cov)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]

    # atan2(dy, dx): screen y grows downward, so flip dy for a natural angle.
    degrees = float(np.degrees(np.arctan2(-axis[1], axis[0])))

    # Map into [-90, 90] so "which way is up the axis" doesn't flip randomly.
    if degrees > 90:
        degrees -= 180
    elif degrees < -90:
        degrees += 180
    return degrees


def bike_visible(frame: np.ndarray, min_pixels: int = 200) -> bool:
    """True when a dark rider silhouette is in the play area (run in progress)."""
    xs, _ys = _silhouette_points(frame)
    return len(xs) >= min_pixels


def find_play_button(frame: np.ndarray) -> tuple[int, int] | None:
    """
    Find the green hexagonal Play button in the lower third.

    Returns (x, y) in frame pixel coords, or None if not found.
    """
    h, w = frame.shape[:2]
    y0, y1 = int(h * 0.58), int(h * 0.88)
    x0, x1 = int(w * 0.18), int(w * 0.82)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    r = crop[:, :, 0].astype(np.int16)
    g = crop[:, :, 1].astype(np.int16)
    b = crop[:, :, 2].astype(np.int16)
    # Vibrant green hex (not yellow "claim", not white icons).
    green = (g > 140) & (g > r + 35) & (g > b + 35) & (r < 120)

    ys, xs = np.nonzero(green)
    if len(xs) < 80:
        return None

    # Largest contiguous-ish cluster via centroid of green mass near horizontal center.
    cx = crop.shape[1] / 2.0
    weights = 1.0 / (1.0 + 0.005 * (xs - cx) ** 2)
    x = int(np.average(xs, weights=weights))
    y = int(np.average(ys, weights=weights))
    return x0 + x, y0 + y


def play_fallback_xy(frame: np.ndarray) -> tuple[int, int]:
    """Fixed spot on the green Play button band (~76% down)."""
    h, w = frame.shape[:2]
    return int(w * 0.5), int(h * 0.76)


def estimate_velocity(angle: float, prev_angle: float | None, dt: float) -> float:
    """Rough angular velocity in degrees per second."""
    if prev_angle is None or dt <= 0:
        return 0.0
    # Handle wrap-ish jumps from PCA sign flips by clamping huge deltas.
    delta = angle - prev_angle
    if abs(delta) > 90:
        return 0.0
    return delta / dt


def _silhouette_points(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = _to_gray(frame)
    h, w = gray.shape
    y0, y1 = int(h * 0.25), int(h * 0.85)
    x0, x1 = int(w * 0.25), int(w * 0.75)
    crop = gray[y0:y1, x0:x1]
    mask = crop < 80
    ys, xs = np.nonzero(mask)
    return xs, ys


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    # ITU-R BT.601 luma
    return (
        0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
    ).astype(np.uint8)
