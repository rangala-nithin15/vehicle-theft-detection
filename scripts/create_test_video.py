"""Utility script to generate a test video using sample frames for testing Phase 1."""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def create_sample_video(
    output_path: Path = PROJECT_ROOT / "data" / "input" / "test.mp4",
    num_frames: int = 60,
    fps: int = 20,
) -> Path:
    """Creates a realistic test video in data/input/ for smoke testing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to find sample image from ultralytics assets if installed
    sample_img = None
    try:
        import ultralytics
        ultralytics_dir = Path(ultralytics.__file__).parent
        asset_bus = ultralytics_dir / "assets" / "bus.jpg"
        if asset_bus.exists():
            sample_img = cv2.imread(str(asset_bus))
    except Exception:
        pass

    # Fallback to realistic synthetic frame if ultralytics asset not found
    if sample_img is None:
        height, width = 480, 640
        sample_img = np.zeros((height, width, 3), dtype=np.uint8)
        # Background road and sky
        sample_img[:200, :] = [200, 150, 100]
        sample_img[200:, :] = [70, 70, 70]

    h, w = sample_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    # Generate slight panning/movement across frames to simulate camera motion
    for i in range(num_frames):
        # Slightly pan across frames
        shift = int(5 * np.sin(i / 10.0))
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        frame = cv2.warpAffine(sample_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        writer.write(frame)

    writer.release()
    print(f"[INFO] Created test video at: {output_path} ({num_frames} frames @ {fps} fps)")
    return output_path


if __name__ == "__main__":
    create_sample_video()
