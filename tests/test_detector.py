"""Unit tests for DetectionConfig, YOLODetector, and VideoProcessor."""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from app.config import DEFAULT_TARGET_CLASSES, DetectionConfig
from app.detector import Detection, YOLODetector
from app.video_processor import VideoProcessor


def test_config_loads() -> None:
    """Test that DetectionConfig loads with valid default settings."""
    cfg = DetectionConfig()
    assert cfg.model_name == "yolov8n.pt"
    assert 0.0 < cfg.confidence_threshold < 1.0
    assert 0.0 < cfg.iou_threshold < 1.0
    assert cfg.device in ("cpu", "cuda", "mps")
    assert isinstance(cfg.target_classes, list)
    assert len(cfg.target_classes) > 0


def test_supported_classes_are_valid() -> None:
    """Test that all required classes from the Phase 1 specification are included."""
    required_classes = {"person", "motorcycle", "bicycle", "car", "truck", "bus"}
    cfg = DetectionConfig()
    configured_set = set(cfg.target_classes)

    # Verify that every required class is present
    for cls in required_classes:
        assert cls in configured_set, f"Required class '{cls}' is missing from target_classes"
        assert cls in DEFAULT_TARGET_CLASSES, f"Required class '{cls}' is missing from DEFAULT_TARGET_CLASSES"


def test_detection_dataclass() -> None:
    """Test the Detection dataclass attributes and helper properties."""
    det = Detection(class_name="car", confidence=0.88, bbox=(10, 20, 100, 150))
    assert det.class_name == "car"
    assert det.confidence == 0.88
    assert det.x1 == 10
    assert det.y1 == 20
    assert det.x2 == 100
    assert det.y2 == 150
    assert det.center == (55, 85)

    data = det.to_dict()
    assert data["class_name"] == "car"
    assert data["confidence"] == 0.88
    assert data["bbox"] == [10, 20, 100, 150]


def test_detector_initialization_with_mock() -> None:
    """Test that YOLODetector initializes without downloading weights when passed a mock model."""
    mock_model = MagicMock()
    detector = YOLODetector(model=mock_model)
    assert detector.model is mock_model
    assert "person" in detector.target_classes
    assert "car" in detector.target_classes


def test_detector_filters_non_target_classes() -> None:
    """Test that YOLODetector only retains configured target classes."""
    # Setup mock YOLO results
    mock_model = MagicMock()
    mock_result = MagicMock()

    # Define mock boxes: 0 is person, 2 is car, 16 is dog (not a target class)
    mock_box_person = MagicMock()
    mock_box_person.cls = [0]
    mock_box_person.conf = [0.85]
    mock_box_person.xyxy = [[50.0, 60.0, 150.0, 200.0]]

    mock_box_dog = MagicMock()
    mock_box_dog.cls = [16]
    mock_box_dog.conf = [0.90]
    mock_box_dog.xyxy = [[200.0, 200.0, 300.0, 300.0]]

    mock_box_car = MagicMock()
    mock_box_car.cls = [2]
    mock_box_car.conf = [0.92]
    mock_box_car.xyxy = [[300.0, 250.0, 500.0, 450.0]]

    mock_result.boxes = [mock_box_person, mock_box_dog, mock_box_car]
    mock_result.names = {0: "person", 2: "car", 16: "dog"}
    mock_model.return_value = [mock_result]

    detector = YOLODetector(model=mock_model)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame)

    # Should only have detected 'person' and 'car', NOT 'dog'
    detected_classes = [d.class_name for d in detections]
    assert "person" in detected_classes
    assert "car" in detected_classes
    assert "dog" not in detected_classes
    assert len(detections) == 2


def test_detector_draw_detections() -> None:
    """Test that draw_detections annotates the image without mutating the original."""
    mock_model = MagicMock()
    detector = YOLODetector(model=mock_model)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [
        Detection(class_name="person", confidence=0.85, bbox=(50, 50, 150, 200)),
        Detection(class_name="car", confidence=0.92, bbox=(200, 200, 400, 400)),
    ]

    annotated = detector.draw_detections(dummy_frame, detections)

    assert annotated.shape == dummy_frame.shape
    # Frame was modified (contains drawn pixels)
    assert np.any(annotated > 0)
    # Original frame remains unchanged
    assert not np.any(dummy_frame > 0)


def test_video_processor_missing_input_raises_error() -> None:
    """Test that passing a non-existent input video file raises FileNotFoundError."""
    mock_model = MagicMock()
    detector = YOLODetector(model=mock_model)
    processor = VideoProcessor(detector=detector)

    non_existent = Path("data/input/does_not_exist_12345.mp4")
    with pytest.raises(FileNotFoundError) as exc_info:
        processor.process_video(input_path=non_existent)

    assert "Input video file not found" in str(exc_info.value)
