"""
Find the Rider window and grab screenshots (macOS).

Needs Screen Recording permission for Terminal/Cursor/Python.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from Quartz import (
    CGDataProviderCopyData,
    CGImageGetBytesPerRow,
    CGImageGetDataProvider,
    CGImageGetHeight,
    CGImageGetWidth,
    CGRectNull,
    CGWindowListCopyWindowInfo,
    CGWindowListCreateImage,
    kCGNullWindowID,
    kCGWindowBounds,
    kCGWindowImageBoundsIgnoreFraming,
    kCGWindowListOptionIncludingWindow,
    kCGWindowListOptionOnScreenOnly,
    kCGWindowName,
    kCGWindowNumber,
    kCGWindowOwnerName,
)

# Ketchapp Rider (iOS app running on Apple Silicon Mac)
APP_NAME = "Rider"


@dataclass
class WindowInfo:
    window_id: int
    left: float
    top: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.left + self.width / 2.0, self.top + self.height / 2.0)

    def frame_to_screen(self, fx: float, fy: float, frame_shape: tuple[int, ...]) -> tuple[float, float]:
        """Map screenshot pixel (fx, fy) → global screen coordinates."""
        fh, fw = frame_shape[0], frame_shape[1]
        x = self.left + (fx / fw) * self.width
        y = self.top + (fy / fh) * self.height
        return x, y


def find_rider_window() -> WindowInfo:
    """Return the largest on-screen Rider window, or raise if not found."""
    options = kCGWindowListOptionOnScreenOnly
    windows = CGWindowListCopyWindowInfo(options, kCGNullWindowID)
    if not windows:
        raise RuntimeError("Could not list windows. Check Screen Recording permission.")

    candidates: list[WindowInfo] = []
    for window in windows:
        owner = window.get(kCGWindowOwnerName, "")
        title = window.get(kCGWindowName, "") or ""
        if owner != APP_NAME and title != APP_NAME:
            continue
        bounds = window.get(kCGWindowBounds)
        if not bounds:
            continue
        width = float(bounds.get("Width", 0))
        height = float(bounds.get("Height", 0))
        if width < 100 or height < 100:
            continue  # skip tiny chrome windows
        candidates.append(
            WindowInfo(
                window_id=int(window[kCGWindowNumber]),
                left=float(bounds["X"]),
                top=float(bounds["Y"]),
                width=width,
                height=height,
            )
        )

    if not candidates:
        raise RuntimeError(
            "Rider window not found. Open /Applications/Rider.app and start a run, "
            "then try again."
        )

    # Prefer the biggest window (the game canvas, not a tiny chrome piece).
    return max(candidates, key=lambda w: w.width * w.height)


def capture_window(info: WindowInfo) -> np.ndarray:
    """Screenshot one window → RGB uint8 array shaped (H, W, 3)."""
    image = CGWindowListCreateImage(
        CGRectNull,
        kCGWindowListOptionIncludingWindow,
        info.window_id,
        kCGWindowImageBoundsIgnoreFraming,
    )
    if image is None:
        raise RuntimeError(
            "Screenshot failed. Grant Screen Recording to your terminal "
            "(System Settings → Privacy & Security → Screen Recording)."
        )

    width = int(CGImageGetWidth(image))
    height = int(CGImageGetHeight(image))
    row_bytes = int(CGImageGetBytesPerRow(image))
    data = CGDataProviderCopyData(CGImageGetDataProvider(image))
    raw = np.frombuffer(data, dtype=np.uint8)

    # macOS screenshots are typically BGRA
    bgra = np.lib.stride_tricks.as_strided(
        raw,
        shape=(height, width, 4),
        strides=(row_bytes, 4, 1),
    ).copy()
    rgb = bgra[:, :, :3][:, :, ::-1]
    return rgb
