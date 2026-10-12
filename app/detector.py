"""YOLO object detector module for detecting persons and vehicles."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.config import DetectionConfig, config as default_config


@dataclass
class Detection:
    """Represents a single detected object in a frame.
    
    Attributes:
        class_name: Label of the detected class (e.g. 'person', 'car').
        confidence: Detection confidence score between 0.0 and 1.0.
        bbox: Bounding box tuple in pixels (x1, y1, x2, y2).
    """

    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def center(self) -> Tuple[int, int]:
        """Returns the center point (cx, cy) of the bounding box."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary representation."""
        return {
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
        }


# Distinct BGR colors for supported classes for clear visualization
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "person": (255, 128, 0),       # Light blue
    "car": (0, 200, 0),            # Green
    "motorcycle": (0, 165, 255),   # Orange
    "bicycle": (0, 255, 255),      # Yellow
    "truck": (200, 0, 200),        # Purple
    "bus": (255, 0, 128),          # Magenta
}
DEFAULT_COLOR: Tuple[int, int, int] = (0, 255, 0)


class YOLODetector:
    """Object detector wrapper around Ultralytics YOLO."""

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        model: Optional[Any] = None,
    ) -> None:
        """Initialize the YOLO detector.
        
        Args:
            config: Detection configuration settings.
            model: Optional pre-loaded or mock YOLO model instance.
        """
        self.config = config or default_config
        self.target_classes = set(self.config.target_classes)

        if model is not None:
            self.model = model
        else:
            # Lazy import ultralytics so non-inference tools or tests can import module quickly
            from ultralytics import YOLO

            model_path = self.config.get_model_path()
            self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run object detection on a single image/video frame.
        
        Args:
            frame: BGR image frame as a NumPy array.
            
        Returns:
            List of Detection objects filtered by configured target classes.
        """
        if frame is None or frame.size == 0:
            return []

        # Run inference
        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            device=self.config.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        first_res = results[0]
        boxes = getattr(first_res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        names = getattr(first_res, "names", {})

        for box in boxes:
            cls_raw = box.cls[0]
            cls_id = int(cls_raw.item() if hasattr(cls_raw, "item") else cls_raw)
            class_name = names.get(cls_id, str(cls_id))

            # Only retain configured classes (person, car, motorcycle, etc.)
            if class_name in self.target_classes:
                conf_raw = box.conf[0]
                conf = float(conf_raw.item() if hasattr(conf_raw, "item") else conf_raw)

                coords_raw = box.xyxy[0]
                if hasattr(coords_raw, "tolist"):
                    coords = coords_raw.tolist()
                elif isinstance(coords_raw, (list, tuple)):
                    coords = list(coords_raw)
                else:
                    coords = [coords_raw[0], coords_raw[1], coords_raw[2], coords_raw[3]]

                x1, y1, x2, y2 = (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))

                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                    )
                )

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> np.ndarray:
        """Annotate frame with bounding boxes and class labels.
        
        Args:
            frame: Input BGR image frame.
            detections: List of detections to draw.
            
        Returns:
            Annotated BGR image frame.
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)

            # Draw bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                thickness=self.config.box_thickness,
            )

            # Label text: class name and confidence percentage
            label = f"{det.class_name} {det.confidence * 100:.1f}%"

            # Compute label background rectangle dimensions
            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                self.config.font_thickness,
            )

            label_y1 = max(0, y1 - text_h - baseline - 4)
            label_y2 = y1

            # Background rectangle for text readability
            cv2.rectangle(
                annotated,
                (x1, label_y1),
                (x1 + text_w + 6, label_y2),
                color,
                -1,
            )

            # Draw white text on top of the label background
            cv2.putText(
                annotated,
                label,
                (x1 + 3, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                (255, 255, 255),
                self.config.font_thickness,
                lineType=cv2.LINE_AA,
            )

        return annotated
