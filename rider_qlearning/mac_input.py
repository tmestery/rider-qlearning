"""
Inject mouse hold / release into the Rider window (macOS).

Needs Accessibility permission for Terminal/Cursor/Python.
Hold = lean forward, release = lean back (same as tapping the screen).
"""

from __future__ import annotations

from Quartz import (
    CGEventCreate,
    CGEventCreateMouseEvent,
    CGEventGetLocation,
    CGEventPost,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseUp,
    kCGHIDEventTap,
    kCGMouseButtonLeft,
)

# Last place we pressed, so emergency_release can let go even if the cursor moved.
_last_xy: tuple[float, float] | None = None
_holding = False


def _click_at(x: float, y: float, down: bool) -> None:
    global _last_xy, _holding
    event_type = kCGEventLeftMouseDown if down else kCGEventLeftMouseUp
    event = CGEventCreateMouseEvent(None, event_type, (x, y), kCGMouseButtonLeft)
    if event is None:
        raise RuntimeError(
            "Could not create mouse event. Grant Accessibility permission "
            "(System Settings → Privacy & Security → Accessibility)."
        )
    CGEventPost(kCGHIDEventTap, event)
    _last_xy = (x, y)
    _holding = down


def hold(x: float, y: float) -> None:
    """Press and keep holding left mouse at (x, y)."""
    _click_at(x, y, down=True)


def release(x: float, y: float) -> None:
    """Release left mouse at (x, y)."""
    _click_at(x, y, down=False)


def emergency_release() -> None:
    """
    Force mouse-up at last click point and current cursor.
    Call this from kill switches / exit handlers.
    """
    global _holding
    points: list[tuple[float, float]] = []
    if _last_xy is not None:
        points.append(_last_xy)
    event = CGEventCreate(None)
    if event is not None:
        loc = CGEventGetLocation(event)
        points.append((float(loc.x), float(loc.y)))

    for x, y in points:
        up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft)
        if up is not None:
            CGEventPost(kCGHIDEventTap, up)
    _holding = False


def is_holding() -> bool:
    return _holding
