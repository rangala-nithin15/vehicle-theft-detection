"""Multi-object tracking module using ByteTrack for persistent ID assignment."""

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from app.config import DetectionConfig, config as default_config
from app.detector import CLASS_COLORS, DEFAULT_COLOR


@dataclass
class TrackedObject:
    """Represents a persistent tracked object across video frames.
    
    Attributes:
        track_id: Unique persistent tracking identifier.
        class_name: Label of the detected class (e.g., 'person', 'car').
        confidence: Detection confidence score between 0.0 and 1.0.
        bbox: Bounding box tuple in pixels (x1, y1, x2, y2).
        frame_idx: Video frame index when this track was observed.
        timestamp: Video timestamp in seconds.
    """

    track_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    frame_idx: int = 0
    timestamp: float = 0.0

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
        """Calculates the center point (cx, cy) of the bounding box."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def label(self) -> str:
        """Formatted label string with class and track ID, e.g. 'Person #4'."""
        return f"{self.class_name.capitalize()} #{self.track_id}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert tracked object state to dictionary."""
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "center": list(self.center),
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 3),
        }


class ObjectTracker:
    """Multi-object tracker utilizing ByteTrack via Ultralytics."""

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        model: Optional[Any] = None,
    ) -> None:
        """Initialize tracker with configuration and YOLO model.
        
        Args:
            config: Detection and tracking configuration.
            model: Optional pre-instantiated YOLO model instance.
        """
        self.config = config or default_config
        self.target_classes = set(self.config.target_classes)

        # Store trajectory history: track_id -> deque of center coordinates
        self.history: Dict[int, Deque[Tuple[int, int]]] = {}
        # Track all unique IDs observed during the session
        self.seen_track_ids: Set[int] = set()

        if model is not None:
            self.model = model
        else:
            from ultralytics import YOLO

            model_path = self.config.get_model_path()
            self.model = YOLO(model_path)

    def track(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        timestamp: float = 0.0,
    ) -> List[TrackedObject]:
        """Run multi-object tracking on a single frame.
        
        Args:
            frame: Input BGR image frame as a NumPy array.
            frame_idx: Current frame index.
            timestamp: Current timestamp in seconds.
            
        Returns:
            List of TrackedObject instances with persistent track IDs.
        """
        if frame is None or frame.size == 0:
            return []

        # Run tracking inference with state persistence
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.config.tracker_type,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            device=self.config.device,
            verbose=False,
        )

        tracks: List[TrackedObject] = []
        if not results:
            return tracks

        first_res = results[0]
        boxes = getattr(first_res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return tracks

        names = getattr(first_res, "names", {})

        for box in boxes:
            # Only consider detections that have been assigned a track ID by the tracker
            if box.id is None:
                continue

            cls_raw = box.cls[0]
            cls_id = int(cls_raw.item() if hasattr(cls_raw, "item") else cls_raw)
            class_name = names.get(cls_id, str(cls_id))

            # Only retain configured classes (person, car, motorcycle, etc.)
            if class_name in self.target_classes:
                id_raw = box.id[0]
                track_id = int(id_raw.item() if hasattr(id_raw, "item") else id_raw)

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

                obj = TrackedObject(
                    track_id=track_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                )
                tracks.append(obj)

                # Record trajectory history
                if track_id not in self.history:
                    self.history[track_id] = deque(maxlen=self.config.track_history_len)
                self.history[track_id].append(obj.center)
                self.seen_track_ids.add(track_id)

        return tracks

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[TrackedObject],
        draw_trail: bool = True,
    ) -> np.ndarray:
        """Annotate frame with bounding boxes, persistent IDs, and trajectory trails.
        
        Args:
            frame: Input BGR image frame.
            tracks: List of tracked objects to draw.
            draw_trail: Whether to draw motion trajectory trails.
            
        Returns:
            Annotated BGR image frame.
        """
        annotated = frame.copy()

        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            color = CLASS_COLORS.get(track.class_name, DEFAULT_COLOR)

            # 1. Draw motion trajectory trail
            if draw_trail and track.track_id in self.history:
                trail_pts = list(self.history[track.track_id])
                for idx in range(1, len(trail_pts)):
                    pt_start = trail_pts[idx - 1]
                    pt_end = trail_pts[idx]
                    # Alpha-fading effect simulated by trail line thickness
                    thickness = max(1, int(idx / len(trail_pts) * 3))
                    cv2.line(annotated, pt_start, pt_end, color, thickness)

            # 2. Draw bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                thickness=self.config.box_thickness,
            )

            # 3. Label text: e.g. "Person #4 87%"
            label = f"{track.label} {track.confidence * 100:.0f}%"

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

            # 4. Draw center point
            cx, cy = track.center
            cv2.circle(annotated, (cx, cy), 3, color, -1)

        return annotated

    def reset(self) -> None:
        """Reset internal trajectory history and seen tracks."""
        self.history.clear()
        self.seen_track_ids.clear()
