"""
Smoke-test Mac capture + angle estimate.

1. Open Rider.app so the game window is visible.
2. Run:  python capture_test.py
3. Check captures/rider_frame.png and the printed angle.

Requires Screen Recording permission.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from rider_qlearning import features, mac_window


def main() -> None:
    out_dir = Path("captures")
    out_dir.mkdir(exist_ok=True)

    info = mac_window.find_rider_window()
    print(
        f"Found Rider window id={info.window_id} "
        f"at ({info.left:.0f},{info.top:.0f}) "
        f"size {info.width:.0f}x{info.height:.0f}"
    )

    frame = mac_window.capture_window(info)
    angle = features.estimate_angle(frame)
    path = out_dir / "rider_frame.png"
    Image.fromarray(frame).save(path)

    print(f"Saved {path}  shape={frame.shape}  estimated_angle={angle:.1f}°")
    print("If the image is black/empty, enable Screen Recording for this terminal.")


if __name__ == "__main__":
    main()
