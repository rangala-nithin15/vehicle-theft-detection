"""Integration tests for the stream_risk_assessment video pipeline generator (Phase 6)."""

from pathlib import Path
from unittest.mock import MagicMock
import cv2
import numpy as np
import pytest

from app.config import DetectionConfig
from app.database import EventDatabase
from app.video_processor import VideoProcessor


@pytest.fixture
def sample_video_file(tmp_path: Path) -> Path:
    """Create a short 5-frame synthetic MP4 video file for pipeline testing."""
    video_path = tmp_path / "synthetic_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (320, 240))
    for i in range(5):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Add a colored square
        cv2.rectangle(frame, (20 + i * 5, 20), (80 + i * 5, 100), (0, 255, 0), -1)
        out.write(frame)
    out.release()
    return video_path


def test_stream_risk_assessment_output_structure(sample_video_file: Path, tmp_path: Path) -> None:
    """Verify that stream_risk_assessment yields valid data structures."""
    db_file = tmp_path / "test_stream.db"
    db = EventDatabase(db_file)
    processor = VideoProcessor(db=db)

    generator = processor.stream_risk_assessment(
        input_path=sample_video_file,
        save_to_db=True,
    )
    yielded_items = list(generator)

    assert len(yielded_items) == 5
    first_item = yielded_items[0]

    expected_keys = {
        "frame_bgr",
        "frame_rgb",
        "frame_idx",
        "total_frames",
        "timestamp",
        "tracks",
        "associations",
        "vehicle_states",
        "assessments",
        "new_alerts",
        "fps",
        "unique_persons",
        "unique_vehicles",
    }
    assert expected_keys.issubset(first_item.keys())
    assert first_item["frame_rgb"].shape == (240, 320, 3)
    assert first_item["frame_idx"] == 1
    assert first_item["total_frames"] == 5


def test_stream_risk_assessment_max_frames(sample_video_file: Path, tmp_path: Path) -> None:
    """Verify that max_frames limits the number of processed frames."""
    processor = VideoProcessor(db=EventDatabase(tmp_path / "stream_max.db"))
    generator = processor.stream_risk_assessment(
        input_path=sample_video_file,
        max_frames=2,
        save_to_db=False,
    )
    items = list(generator)
    assert len(items) == 2


def test_stream_risk_assessment_file_not_found(tmp_path: Path) -> None:
    """Verify FileNotFoundError when non-existent video path is provided."""
    processor = VideoProcessor()
    with pytest.raises(FileNotFoundError):
        list(processor.stream_risk_assessment(tmp_path / "non_existent.mp4"))
