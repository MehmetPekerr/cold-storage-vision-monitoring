"""Generate a fully synthetic cold-storage monitoring demonstration.

No source images, camera streams, model weights, or company data are required.
"""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "synthetic"
SIZE = (640, 360)


def draw_frame(index: int) -> np.ndarray:
    width, height = SIZE
    frame = np.full((height, width, 3), (38, 45, 54), dtype=np.uint8)

    # Cartoon cold-room walls and floor.
    cv2.rectangle(frame, (40, 35), (width - 40, height - 45), (88, 98, 108), -1)
    cv2.line(frame, (40, 230), (width - 40, 230), (170, 180, 190), 2)
    cv2.rectangle(frame, (90, 75), (230, 230), (150, 160, 170), 3)

    door_open = (index // 20) % 2 == 1
    door_color = (35, 185, 225) if door_open else (55, 190, 75)
    door_right = 220 if door_open else 205
    cv2.rectangle(frame, (95, 78), (door_right, 227), door_color, 4)
    cv2.putText(
        frame,
        "OPEN" if door_open else "CLOSED",
        (105, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        door_color,
        2,
    )

    corridor = np.array([[260, 225], [535, 225], [590, 315], [205, 315]], np.int32)
    occupied = (index // 10) % 3 != 0
    corridor_color = (35, 190, 220) if occupied else (65, 200, 80)
    cv2.polylines(frame, [corridor], True, corridor_color, 3)
    if occupied:
        cv2.rectangle(frame, (330, 250), (420, 300), (165, 105, 55), -1)
        cv2.line(frame, (375, 250), (375, 300), (220, 220, 220), 3)

    occupancy = 42.0 if occupied else 0.0
    cv2.putText(frame, f"DOOR: {'OPEN' if door_open else 'CLOSED'}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, door_color, 2)
    cv2.putText(frame, f"FLOOR: {'OCCUPIED' if occupied else 'EMPTY'} ({occupancy:.0f}%)",
                (20, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                corridor_color, 2)
    cv2.putText(frame, "SYNTHETIC DEMO", (width - 205, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)
    return frame


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video_path = OUTPUT_DIR / "synthetic_monitor_demo.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        SIZE,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create demo video: {video_path}")

    try:
        for index in range(80):
            frame = draw_frame(index)
            writer.write(frame)
            if index in (0, 40, 79):
                cv2.imwrite(str(OUTPUT_DIR / f"synthetic_frame_{index:03d}.png"), frame)
    finally:
        writer.release()
    print(f"Generated synthetic demo: {video_path}")


if __name__ == "__main__":
    main()
