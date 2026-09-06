"""
Inject mouse hold / release into the Rider window (macOS).

Needs Accessibility permission for Terminal/Cursor/Python.
Hold = lean forward, release = lean back (same as tapping the screen).
"""

from __future__ import annotations

from Quartz import (
    CGEventCreateMouseEvent,
    CGEventPost,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseUp,
    kCGHIDEventTap,
    kCGMouseButtonLeft,
)


def _click_at(x: float, y: float, down: bool) -> None:
    event_type = kCGEventLeftMouseDown if down else kCGEventLeftMouseUp
    event = CGEventCreateMouseEvent(None, event_type, (x, y), kCGMouseButtonLeft)
    if event is None:
        raise RuntimeError(
            "Could not create mouse event. Grant Accessibility permission "
            "(System Settings → Privacy & Security → Accessibility)."
        )
    CGEventPost(kCGHIDEventTap, event)


def hold(x: float, y: float) -> None:
    """Press and keep holding left mouse at (x, y)."""
    _click_at(x, y, down=True)


def release(x: float, y: float) -> None:
    """Release left mouse at (x, y)."""
    _click_at(x, y, down=False)
