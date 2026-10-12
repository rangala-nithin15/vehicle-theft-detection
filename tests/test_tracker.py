"""Unit tests for TrackedObject and ObjectTracker (Phase 2)."""

from unittest.mock import MagicMock
import numpy as np
import pytest

from app.config import DetectionConfig
from app.tracker import ObjectTracker, TrackedObject


def test_tracked_object_representation() -> None:
    """Test TrackedObject attributes, properties, and dictionary conversion."""
    obj = TrackedObject(
        track_id=4,
        class_name="person",
        confidence=0.8912,
        bbox=(50, 100, 150, 300),
        frame_idx=12,
        timestamp=0.48,
    )

    assert obj.track_id == 4
    assert obj.class_name == "person"
    assert obj.confidence == pytest.approx(0.8912, rel=1e-3)
    assert obj.bbox == (50, 100, 150, 300)
    assert obj.frame_idx == 12
    assert obj.timestamp == pytest.approx(0.48, rel=1e-3)

    # Label formatting
    assert obj.label == "Person #4"

    # Coordinate properties
    assert obj.x1 == 50
    assert obj.y1 == 100
    assert obj.x2 == 150
    assert obj.y2 == 300

    # Serialization
    data = obj.to_dict()
    assert data["track_id"] == 4
    assert data["class_name"] == "person"
    assert data["center"] == [100, 200]
    assert data["bbox"] == [50, 100, 150, 300]


def test_center_point_calculation() -> None:
    """Test accurate center point calculation across various bounding boxes."""
    # Test case 1
    obj1 = TrackedObject(
        track_id=1,
        class_name="car",
        confidence=0.95,
        bbox=(100, 200, 300, 400),
    )
    assert obj1.center == (200, 300)

    # Test case 2: odd dimensions
    obj2 = TrackedObject(
        track_id=2,
        class_name="motorcycle",
        confidence=0.91,
        bbox=(15, 25, 46, 76),
    )
    # (15 + 46) // 2 = 30, (25 + 76) // 2 = 50
    assert obj2.center == (30, 50)


def test_tracker_parses_valid_track_ids_and_filters() -> None:
    """Test that ObjectTracker extracts valid track IDs and ignores unassigned/unsupported classes."""
    mock_model = MagicMock()
    mock_result = MagicMock()

    # Box 1: Confirmed person with track ID 1
    b1 = MagicMock()
    b1.id = [1]
    b1.cls = [0]
    b1.conf = [0.88]
    b1.xyxy = [[50.0, 60.0, 120.0, 180.0]]

    # Box 2: Confirmed car with track ID 7
    b2 = MagicMock()
    b2.id = [7]
    b2.cls = [2]
    b2.conf = [0.93]
    b2.xyxy = [[200.0, 200.0, 450.0, 400.0]]

    # Box 3: Detection with NO track ID yet (unassigned)
    b3 = MagicMock()
    b3.id = None
    b3.cls = [0]
    b3.conf = [0.45]
    b3.xyxy = [[10.0, 10.0, 50.0, 50.0]]

    # Box 4: Non-target class (e.g. dog: 16) with track ID 9
    b4 = MagicMock()
    b4.id = [9]
    b4.cls = [16]
    b4.conf = [0.90]
    b4.xyxy = [[300.0, 300.0, 350.0, 350.0]]

    mock_result.boxes = [b1, b2, b3, b4]
    mock_result.names = {0: "person", 2: "car", 16: "dog"}
    mock_model.track.return_value = [mock_result]

    cfg = DetectionConfig(track_history_len=10)
    tracker = ObjectTracker(config=cfg, model=mock_model)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracks = tracker.track(dummy_frame, frame_idx=1, timestamp=0.04)

    # Should only retain person (ID 1) and car (ID 7)
    assert len(tracks) == 2
    assert tracks[0].track_id == 1
    assert tracks[0].label == "Person #1"
    assert tracks[1].track_id == 7
    assert tracks[1].label == "Car #7"

    # Verify tracker internal records
    assert 1 in tracker.seen_track_ids
    assert 7 in tracker.seen_track_ids
    assert 9 not in tracker.seen_track_ids
    assert len(tracker.history[1]) == 1
    assert len(tracker.history[7]) == 1


def test_tracker_draw_tracks_and_trails() -> None:
    """Test that drawing tracks and motion trails renders without mutating original image."""
    mock_model = MagicMock()
    tracker = ObjectTracker(model=mock_model)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracks = [
        TrackedObject(track_id=3, class_name="car", confidence=0.92, bbox=(100, 100, 300, 300)),
        TrackedObject(track_id=8, class_name="person", confidence=0.85, bbox=(400, 100, 480, 320)),
    ]

    # Pre-populate some history for trail rendering
    from collections import deque
    tracker.history[3] = deque([(180, 180), (190, 190), (200, 200)], maxlen=30)

    annotated = tracker.draw_tracks(dummy_frame, tracks, draw_trail=True)

    assert annotated.shape == dummy_frame.shape
    # Frame was modified with annotations
    assert np.any(annotated > 0)
    # Original frame remains unchanged
    assert not np.any(dummy_frame > 0)


def test_tracker_reset() -> None:
    """Test that reset clears trajectories and seen track IDs."""
    mock_model = MagicMock()
    tracker = ObjectTracker(model=mock_model)

    from collections import deque
    tracker.history[1] = deque([(10, 10)], maxlen=10)
    tracker.seen_track_ids.add(1)

    assert len(tracker.history) == 1
    assert len(tracker.seen_track_ids) == 1

    tracker.reset()

    assert len(tracker.history) == 0
    assert len(tracker.seen_track_ids) == 0


def test_tracker_empty_frame_returns_empty() -> None:
    """Test that passing an empty frame to track returns an empty list without error."""
    mock_model = MagicMock()
    tracker = ObjectTracker(model=mock_model)

    assert tracker.track(None) == []
    assert tracker.track(np.empty((0, 0, 3), dtype=np.uint8)) == []
