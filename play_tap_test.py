"""
Debug Play-button tapping.

1. Open Rider on the menu / game-over screen (Play visible).
2. Run:  python play_tap_test.py
3. Watch the cursor jump to the target, then check captures/play_tap_*.png
   (red crosshair = where we clicked).

Kill switches are not armed — move the mouse yourself if needed.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from rider_qlearning import features, mac_input, mac_window


def _annotate(frame, fx: int, fy: int, path: Path) -> None:
    img = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(img)
    r = 18
    draw.ellipse((fx - r, fy - r, fx + r, fy + r), outline=(255, 0, 0), width=3)
    draw.line((fx - 28, fy, fx + 28, fy), fill=(255, 0, 0), width=2)
    draw.line((fx, fy - 28, fx, fy + 28), fill=(255, 0, 0), width=2)
    # mark lower-third band
    h, w = frame.shape[:2]
    draw.rectangle((0, int(h * 0.62), w - 1, int(h * 0.95)), outline=(0, 255, 0), width=2)
    img.save(path)


def main() -> None:
    out = Path("captures")
    out.mkdir(exist_ok=True)

    mac_input.activate_rider()
    info = mac_window.find_rider_window()
    print(
        f"Rider window id={info.window_id} "
        f"({info.left:.0f},{info.top:.0f}) {info.width:.0f}x{info.height:.0f}"
    )

    frame = mac_window.capture_window(info)
    found = features.find_play_button(frame)
    if found is None:
        fx, fy = features.play_fallback_xy(frame)
        print(f"Play not detected — using fallback frame=({fx},{fy})")
    else:
        fx, fy = found
        print(f"Play detected at frame=({fx},{fy})")

    before = out / "play_tap_before.png"
    _annotate(frame, fx, fy, before)
    print(f"Saved {before} (green box = search band, red = tap target)")

    x, y = info.frame_to_screen(fx, fy, frame.shape)
    print(f"Moving cursor + tapping screen=({x:.0f},{y:.0f}) in 1s…")
    import time

    time.sleep(1.0)
    mac_input.tap(x, y, hold_seconds=0.2)
    time.sleep(0.8)

    after_frame = mac_window.capture_window(info)
    after = out / "play_tap_after.png"
    Image.fromarray(after_frame).save(after)
    print(f"Saved {after}")
    print(f"bike_visible after tap? {features.bike_visible(after_frame)}")


if __name__ == "__main__":
    main()
