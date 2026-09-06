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
    gray = _to_gray(frame)
    h, w = gray.shape

    # Focus on the center where the bike sits.
    y0, y1 = int(h * 0.25), int(h * 0.85)
    x0, x1 = int(w * 0.25), int(w * 0.75)
    crop = gray[y0:y1, x0:x1]

    # Silhouette is dark against a brighter background.
    mask = crop < 80
    ys, xs = np.nonzero(mask)
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


def estimate_velocity(angle: float, prev_angle: float | None, dt: float) -> float:
    """Rough angular velocity in degrees per second."""
    if prev_angle is None or dt <= 0:
        return 0.0
    # Handle wrap-ish jumps from PCA sign flips by clamping huge deltas.
    delta = angle - prev_angle
    if abs(delta) > 90:
        return 0.0
    return delta / dt


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    # ITU-R BT.601 luma
    return (
        0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
    ).astype(np.uint8)
