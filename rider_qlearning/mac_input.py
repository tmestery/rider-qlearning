"""
Inject mouse hold / release / tap into the Rider window (macOS).

Needs Accessibility permission for Terminal/Cursor/Python.
Hold = lean forward, release = lean back (same as tapping the screen).
"""

from __future__ import annotations

import subprocess
import time

from Quartz import (
    CGEventCreate,
    CGEventCreateMouseEvent,
    CGEventGetLocation,
    CGEventPost,
    CGEventSetIntegerValueField,
    CGWarpMouseCursorPosition,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseUp,
    kCGEventMouseMoved,
    kCGHIDEventTap,
    kCGMouseButtonLeft,
    kCGMouseEventClickState,
    kCGSessionEventTap,
)

# Last place we pressed, so emergency_release can let go even if the cursor moved.
_last_xy: tuple[float, float] | None = None
_holding = False


def activate_rider() -> None:
    """Bring Rider to the foreground so it receives clicks."""
    subprocess.run(
        ["osascript", "-e", 'tell application "Rider" to activate'],
        check=False,
        capture_output=True,
    )
    time.sleep(0.35)


def _post_mouse(event_type: int, x: float, y: float, click_state: int = 1) -> None:
    event = CGEventCreateMouseEvent(None, event_type, (x, y), kCGMouseButtonLeft)
    if event is None:
        raise RuntimeError(
            "Could not create mouse event. Grant Accessibility permission "
            "(System Settings → Privacy & Security → Accessibility)."
        )
    if event_type in (kCGEventLeftMouseDown, kCGEventLeftMouseUp):
        CGEventSetIntegerValueField(event, kCGMouseEventClickState, click_state)
    # Post to both taps — iOS-on-Mac apps are picky.
    CGEventPost(kCGHIDEventTap, event)
    CGEventPost(kCGSessionEventTap, event)


def move_to(x: float, y: float) -> None:
    """Warp + move event so the cursor is visibly on the target."""
    CGWarpMouseCursorPosition((x, y))
    _post_mouse(kCGEventMouseMoved, x, y)


def _click_at(x: float, y: float, down: bool) -> None:
    global _last_xy, _holding
    move_to(x, y)
    time.sleep(0.03)
    _post_mouse(kCGEventLeftMouseDown if down else kCGEventLeftMouseUp, x, y)
    _last_xy = (x, y)
    _holding = down


def hold(x: float, y: float) -> None:
    """Press and keep holding left mouse at (x, y)."""
    _click_at(x, y, down=True)


def release(x: float, y: float) -> None:
    """Release left mouse at (x, y)."""
    _click_at(x, y, down=False)


def _applescript_click(x: float, y: float) -> None:
    """Fallback click via System Events (sometimes works when CGEvent does not)."""
    script = f'''
    tell application "Rider" to activate
    delay 0.2
    tell application "System Events"
        click at {{{int(round(x))}, {int(round(y))}}}
    end tell
    '''
    subprocess.run(["osascript", "-e", script], check=False, capture_output=True)


def tap(x: float, y: float, hold_seconds: float = 0.18) -> None:
    """
    Full click: warp cursor → CG down/up → AppleScript click fallback.
    """
    global _last_xy, _holding
    activate_rider()
    move_to(x, y)
    time.sleep(0.08)
    _post_mouse(kCGEventLeftMouseDown, x, y)
    _holding = True
    _last_xy = (x, y)
    time.sleep(hold_seconds)
    _post_mouse(kCGEventLeftMouseUp, x, y)
    _holding = False
    # Second chance for stubborn iOS-on-Mac windows.
    time.sleep(0.05)
    _applescript_click(x, y)


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
        _post_mouse(kCGEventLeftMouseUp, x, y)
    _holding = False


def is_holding() -> bool:
    return _holding
