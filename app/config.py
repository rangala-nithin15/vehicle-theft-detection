"""Configuration module for vehicle theft detection system."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
from dotenv import load_dotenv
import yaml

# Load environment variables if .env is present
load_dotenv()

# Root directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent

# Core target classes required for detection (COCO class names)
DEFAULT_TARGET_CLASSES: List[str] = [
    "person",
    "motorcycle",
    "bicycle",
    "car",
    "truck",
    "bus",
]

DEFAULT_PARKING_ZONE: List[List[int]] = [
    [50, 50],
    [1200, 50],
    [1200, 700],
    [50, 700],
]


@dataclass
class DetectionConfig:
    """Central configuration for YOLO object detection, tracking, association, and behavior analysis.
    
    All constants, thresholds, zones, and paths are defined here to keep
    the rest of the codebase clean and modular.
    """

    # Model settings
    model_name: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
    iou_threshold: float = float(os.getenv("IOU_THRESHOLD", "0.45"))
    device: str = os.getenv("YOLO_DEVICE", "cpu")  # 'cpu', 'cuda', or 'mps'

    # Filtered classes
    target_classes: List[str] = field(
        default_factory=lambda: list(DEFAULT_TARGET_CLASSES)
    )

    # Standard directory paths
    models_dir: Path = BASE_DIR / "models"
    input_dir: Path = BASE_DIR / "data" / "input"
    output_dir: Path = BASE_DIR / "data" / "output"

    # Tracking settings (Phase 2)
    tracker_type: str = os.getenv("TRACKER_TYPE", "bytetrack.yaml")  # 'bytetrack.yaml' or 'botsort.yaml'
    track_history_len: int = int(os.getenv("TRACK_HISTORY_LEN", "30"))

    # Parking zone polygon points (Phase 3)
    parking_zone_points: List[List[int]] = field(
        default_factory=lambda: list(DEFAULT_PARKING_ZONE)
    )

    # Person-vehicle association settings (Phase 3)
    interaction_margin_px: int = int(os.getenv("INTERACTION_MARGIN_PX", "50"))
    max_association_distance_px: float = float(os.getenv("MAX_ASSOCIATION_DISTANCE_PX", "150.0"))
    min_persistence_frames: int = int(os.getenv("MIN_PERSISTENCE_FRAMES", "5"))
    min_persistence_seconds: float = float(os.getenv("MIN_PERSISTENCE_SECONDS", "0.5"))
    association_cooldown_frames: int = int(os.getenv("ASSOCIATION_COOLDOWN_FRAMES", "10"))

    # Temporal Behavior State Machine settings (Phase 4)
    movement_threshold_px: float = float(os.getenv("MOVEMENT_THRESHOLD_PX", "18.0"))
    movement_window_frames: int = int(os.getenv("MOVEMENT_WINDOW_FRAMES", "10"))
    movement_consecutive_frames: int = int(os.getenv("MOVEMENT_CONSECUTIVE_FRAMES", "4"))
    approaching_distance_px: float = float(os.getenv("APPROACHING_DISTANCE_PX", "250.0"))
    near_distance_px: float = float(os.getenv("NEAR_DISTANCE_PX", "120.0"))
    interaction_duration_sec: float = float(os.getenv("INTERACTION_DURATION_SEC", "1.0"))
    access_attempt_duration_sec: float = float(os.getenv("ACCESS_ATTEMPT_DURATION_SEC", "2.5"))
    stationary_duration_sec: float = float(os.getenv("STATIONARY_DURATION_SEC", "1.5"))

    # Authorization and Risk Engine settings (Phase 5)
    authorization_config_path: Path = BASE_DIR / "configs" / "authorization.yaml"
    weight_unknown_person: int = int(os.getenv("WEIGHT_UNKNOWN_PERSON", "20"))
    weight_approach: int = int(os.getenv("WEIGHT_APPROACH", "10"))
    weight_interaction: int = int(os.getenv("WEIGHT_INTERACTION", "20"))
    weight_access_attempt: int = int(os.getenv("WEIGHT_ACCESS_ATTEMPT", "20"))
    weight_movement: int = int(os.getenv("WEIGHT_MOVEMENT", "30"))
    weight_exit: int = int(os.getenv("WEIGHT_EXIT", "40"))
    medium_risk_threshold: int = int(os.getenv("MEDIUM_RISK_THRESHOLD", "40"))
    high_risk_threshold: int = int(os.getenv("HIGH_RISK_THRESHOLD", "70"))
    snapshots_dir: Path = BASE_DIR / "data" / "output" / "snapshots"
    db_path: Path = BASE_DIR / "data" / "output" / "surveillance.db"

    # Visualization styling
    box_thickness: int = 2
    font_scale: float = 0.5
    font_thickness: int = 1

    def get_model_path(self) -> str:
        """Return path to local model weights if present, otherwise model identifier."""
        local_model = self.models_dir / self.model_name
        if local_model.exists():
            return str(local_model)
        # Default to model name (Ultralytics auto-downloads to cache/models if not found)
        return self.model_name

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "DetectionConfig":
        """Load configuration overrides from a YAML file."""
        instance = cls()
        if not yaml_path.exists():
            return instance

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Parking zone points
        pz = data.get("parking_zone", {})
        if "points" in pz:
            instance.parking_zone_points = pz["points"]

        # Association parameters
        assoc = data.get("association", {})
        if "interaction_margin_px" in assoc:
            instance.interaction_margin_px = int(assoc["interaction_margin_px"])
        if "max_distance_px" in assoc:
            instance.max_association_distance_px = float(assoc["max_distance_px"])
        if "min_persistence_frames" in assoc:
            instance.min_persistence_frames = int(assoc["min_persistence_frames"])
        if "min_persistence_seconds" in assoc:
            instance.min_persistence_seconds = float(assoc["min_persistence_seconds"])
        if "cooldown_frames" in assoc:
            instance.association_cooldown_frames = int(assoc["cooldown_frames"])

        # Behavior parameters
        beh = data.get("behavior", {})
        if "movement_threshold_px" in beh:
            instance.movement_threshold_px = float(beh["movement_threshold_px"])
        if "movement_window_frames" in beh:
            instance.movement_window_frames = int(beh["movement_window_frames"])
        if "movement_consecutive_frames" in beh:
            instance.movement_consecutive_frames = int(beh["movement_consecutive_frames"])
        if "approaching_distance_px" in beh:
            instance.approaching_distance_px = float(beh["approaching_distance_px"])
        if "near_distance_px" in beh:
            instance.near_distance_px = float(beh["near_distance_px"])
        if "interaction_duration_sec" in beh:
            instance.interaction_duration_sec = float(beh["interaction_duration_sec"])
        if "access_attempt_duration_sec" in beh:
            instance.access_attempt_duration_sec = float(beh["access_attempt_duration_sec"])
        if "stationary_duration_sec" in beh:
            instance.stationary_duration_sec = float(beh["stationary_duration_sec"])

        # Risk weights and thresholds
        rw = data.get("risk_weights", {})
        if "unknown_person" in rw:
            instance.weight_unknown_person = int(rw["unknown_person"])
        if "approach" in rw:
            instance.weight_approach = int(rw["approach"])
        if "persistent_interaction" in rw:
            instance.weight_interaction = int(rw["persistent_interaction"])
        if "access_attempt" in rw:
            instance.weight_access_attempt = int(rw["access_attempt"])
        if "vehicle_movement" in rw:
            instance.weight_movement = int(rw["vehicle_movement"])
        if "exit_crossing" in rw:
            instance.weight_exit = int(rw["exit_crossing"])

        rt = data.get("risk_thresholds", {})
        if "medium" in rt:
            instance.medium_risk_threshold = int(rt["medium"])
        if "high" in rt:
            instance.high_risk_threshold = int(rt["high"])

        return instance


# Default singleton config instance
config = DetectionConfig()
