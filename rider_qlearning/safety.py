"""
Kill switches so you can take the Mac back while train_mac is running.

Armed when training starts. Any of these aborts and releases the mouse:
  - Move the cursor into the top-left corner (about 8px)
  - Hold Escape
  - Ctrl+C (SIGINT)

Always releases the left mouse button on exit.
"""

from __future__ import annotations

import atexit
import signal
import time

from Quartz import (
    CGEventCreate,
    CGEventGetLocation,
    CGEventSourceKeyState,
    kCGEventSourceStateHIDSystemState,
)

from rider_qlearning import mac_input

# Escape key code on macOS
_ESCAPE = 53
_CORNER_PX = 8.0


class AbortTraining(Exception):
    """Raised when a kill switch fires."""


_armed = False
_abort_requested = False


def arm() -> None:
    """Turn on kill switches and exit hooks. Call once before training."""
    global _armed, _abort_requested
    _armed = True
    _abort_requested = False
    atexit.register(shutdown)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)


def shutdown() -> None:
    """Release mouse and disarm. Safe to call more than once."""
    global _armed
    mac_input.emergency_release()
    _armed = False


def request_abort() -> None:
    global _abort_requested
    _abort_requested = True


def _on_signal(signum: int, _frame: object) -> None:
    print("\n*** Kill switch: signal received — releasing mouse ***")
    request_abort()
    shutdown()
    raise SystemExit(128 + signum)


def mouse_in_abort_corner() -> bool:
    event = CGEventCreate(None)
    if event is None:
        return False
    loc = CGEventGetLocation(event)
    return loc.x <= _CORNER_PX and loc.y <= _CORNER_PX


def escape_held() -> bool:
    return bool(CGEventSourceKeyState(kCGEventSourceStateHIDSystemState, _ESCAPE))


def check() -> None:
    """Call often during training. Raises AbortTraining if a kill switch fired."""
    if not _armed:
        return
    if _abort_requested or mouse_in_abort_corner() or escape_held():
        reason = "abort requested"
        if mouse_in_abort_corner():
            reason = "mouse in top-left corner"
        elif escape_held():
            reason = "Escape held"
        print(f"\n*** Kill switch: {reason} — releasing mouse ***")
        shutdown()
        raise AbortTraining(reason)


def sleep(seconds: float, slice_seconds: float = 0.05) -> None:
    """time.sleep that checks kill switches every slice."""
    end = time.time() + seconds
    while True:
        check()
        remaining = end - time.time()
        if remaining <= 0:
            return
        time.sleep(min(slice_seconds, remaining))
