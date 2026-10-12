"""Person-Vehicle association engine combining spatial proximity, interaction zones, and temporal persistence."""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from app.config import DetectionConfig, config as default_config
from app.tracker import TrackedObject
from app.zone import ParkingZone, get_vehicle_interaction_zone, is_point_in_rect


@dataclass
class PersonVehicleAssociation:
    """Represents a spatial and temporal association between a person and a vehicle.
    
    Attributes:
        person_id: Persistent track ID of the person.
        vehicle_id: Persistent track ID of the vehicle.
        vehicle_class: Class label of the vehicle (e.g. 'motorcycle', 'car').
        distance: Distance in pixels from person ground position to vehicle bounding box.
        center_distance: Distance between person center and vehicle center.
        confidence: Association confidence score between 0.0 and 1.0.
        first_seen: Timestamp (in seconds) when this pair was first detected interacting.
        last_seen: Timestamp (in seconds) when this pair was most recently observed.
        duration: Interaction duration in seconds (last_seen - first_seen).
        frame_count: Number of consecutive/rolling frames this association persisted.
        is_persistent: True if interaction meets minimum frames or duration threshold.
        vehicle_in_parking_zone: True if vehicle is located inside the parking zone.
        in_interaction_zone: True if person is inside vehicle expanded interaction box.
    """

    person_id: int
    vehicle_id: int
    vehicle_class: str
    distance: float
    center_distance: float
    confidence: float
    first_seen: float
    last_seen: float
    duration: float
    frame_count: int
    is_persistent: bool
    vehicle_in_parking_zone: bool
    in_interaction_zone: bool

    @property
    def label(self) -> str:
        """Formatted human-readable relationship label."""
        return f"Person #{self.person_id} -> {self.vehicle_class.capitalize()} #{self.vehicle_id}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert association record to dictionary."""
        return {
            "person_id": self.person_id,
            "vehicle_id": self.vehicle_id,
            "vehicle_class": self.vehicle_class,
            "distance": round(self.distance, 2),
            "center_distance": round(self.center_distance, 2),
            "confidence": round(self.confidence, 4),
            "first_seen": round(self.first_seen, 2),
            "last_seen": round(self.last_seen, 2),
            "duration": round(self.duration, 2),
            "frame_count": self.frame_count,
            "is_persistent": self.is_persistent,
            "vehicle_in_parking_zone": self.vehicle_in_parking_zone,
            "in_interaction_zone": self.in_interaction_zone,
        }


def euclidean_distance(pt1: Tuple[int, int], pt2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two 2D points."""
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])


def point_to_rect_distance(
    point: Tuple[int, int],
    rect: Tuple[int, int, int, int],
) -> float:
    """Calculate the shortest Euclidean distance from a point to a rectangle.
    
    If the point is inside the rectangle, the distance is 0.0.
    """
    px, py = point
    rx1, ry1, rx2, ry2 = rect

    # Distance along x-axis
    dx = 0.0
    if px < rx1:
        dx = rx1 - px
    elif px > rx2:
        dx = px - rx2

    # Distance along y-axis
    dy = 0.0
    if py < ry1:
        dy = ry1 - py
    elif py > ry2:
        dy = py - ry2

    return math.hypot(dx, dy)


class AssociationEngine:
    """Reasoning engine associating tracked persons with nearby tracked vehicles over time."""

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        parking_zone: Optional[ParkingZone] = None,
    ) -> None:
        """Initialize the association engine.
        
        Args:
            config: Detection, tracking, and association configuration.
            parking_zone: Optional ParkingZone polygon instance.
        """
        self.config = config or default_config
        self.parking_zone = parking_zone or ParkingZone(self.config.parking_zone_points)

        # Internal state tracking active candidate associations: (person_id, vehicle_id) -> state dict
        self._active_associations: Dict[Tuple[int, int], Dict[str, Any]] = {}
        # Track last frame index an association was seen for cooldown cleanup
        self._last_seen_frame: Dict[Tuple[int, int], int] = {}

    def update(
        self,
        tracked_objects: List[TrackedObject],
        frame_idx: int = 0,
        timestamp: float = 0.0,
    ) -> List[PersonVehicleAssociation]:
        """Update person-vehicle associations for the current frame.
        
        Args:
            tracked_objects: List of active TrackedObjects from the tracker.
            frame_idx: Current frame sequence index.
            timestamp: Video timestamp in seconds.
            
        Returns:
            List of PersonVehicleAssociation objects active in the current frame.
        """
        persons = [obj for obj in tracked_objects if obj.class_name == "person"]
        vehicles = [
            obj for obj in tracked_objects
            if obj.class_name in {"car", "motorcycle", "bicycle", "truck", "bus"}
        ]

        current_frame_pairs: Set[Tuple[int, int]] = set()
        current_associations: List[PersonVehicleAssociation] = []

        # Evaluate all person-vehicle pairs in the current frame
        for person in persons:
            # Person bottom-center represents feet position on the ground plane
            person_bottom_center = (person.center[0], person.y2)

            for vehicle in vehicles:
                # 1. Check whether vehicle is inside the monitored parking zone
                vehicle_in_zone = self.parking_zone.contains_box(vehicle.bbox, check_bottom_center=True)

                # 2. Check vehicle's expanded interaction zone
                interaction_rect = get_vehicle_interaction_zone(
                    vehicle.bbox,
                    margin_px=self.config.interaction_margin_px,
                )
                in_interaction_zone = is_point_in_rect(person_bottom_center, interaction_rect) or is_point_in_rect(person.center, interaction_rect)

                # 3. Calculate distances
                dist_to_bbox = point_to_rect_distance(person_bottom_center, vehicle.bbox)
                center_dist = euclidean_distance(person.center, vehicle.center)

                # Check proximity threshold or interaction zone inclusion
                if in_interaction_zone or dist_to_bbox <= self.config.max_association_distance_px:
                    pair_key = (person.track_id, vehicle.track_id)
                    current_frame_pairs.add(pair_key)
                    self._last_seen_frame[pair_key] = frame_idx

                    # Initialize or update temporal state
                    if pair_key not in self._active_associations:
                        self._active_associations[pair_key] = {
                            "first_seen": timestamp,
                            "last_seen": timestamp,
                            "frame_count": 1,
                        }
                    else:
                        state = self._active_associations[pair_key]
                        state["last_seen"] = timestamp
                        state["frame_count"] += 1

                    state = self._active_associations[pair_key]
                    duration = max(0.0, state["last_seen"] - state["first_seen"])
                    frame_count = state["frame_count"]

                    # Persistence requirement: minimum frames or minimum duration
                    is_persistent = (
                        frame_count >= self.config.min_persistence_frames
                        or duration >= self.config.min_persistence_seconds
                    )

                    # Compute association confidence score
                    base_conf = 0.5 if in_interaction_zone else 0.3
                    proximity_bonus = (
                        max(0.0, 1.0 - (dist_to_bbox / max(1.0, self.config.max_association_distance_px))) * 0.4
                    )
                    persistence_bonus = min(duration / 3.0, 1.0) * 0.1
                    total_conf = min(1.0, base_conf + proximity_bonus + persistence_bonus)

                    assoc = PersonVehicleAssociation(
                        person_id=person.track_id,
                        vehicle_id=vehicle.track_id,
                        vehicle_class=vehicle.class_name,
                        distance=dist_to_bbox,
                        center_distance=center_dist,
                        confidence=total_conf,
                        first_seen=state["first_seen"],
                        last_seen=state["last_seen"],
                        duration=duration,
                        frame_count=frame_count,
                        is_persistent=is_persistent,
                        vehicle_in_parking_zone=vehicle_in_zone,
                        in_interaction_zone=in_interaction_zone,
                    )
                    current_associations.append(assoc)

        # Clean up stale associations that haven't been observed within cooldown frames
        stale_keys = [
            key for key, last_frame in self._last_seen_frame.items()
            if (frame_idx - last_frame) > self.config.association_cooldown_frames
        ]
        for key in stale_keys:
            self._active_associations.pop(key, None)
            self._last_seen_frame.pop(key, None)

        return current_associations

    def draw_associations(
        self,
        frame: np.ndarray,
        associations: List[PersonVehicleAssociation],
        tracked_objects: List[TrackedObject],
    ) -> np.ndarray:
        """Annotate frame with connecting lines and interaction tags between associated pairs.
        
        Args:
            frame: Input BGR image frame.
            associations: List of current PersonVehicleAssociation instances.
            tracked_objects: List of current active TrackedObjects.
            
        Returns:
            Annotated BGR image frame.
        """
        annotated = frame.copy()
        obj_map = {obj.track_id: obj for obj in tracked_objects}

        for assoc in associations:
            person = obj_map.get(assoc.person_id)
            vehicle = obj_map.get(assoc.vehicle_id)
            if not person or not vehicle:
                continue

            # Visual styling differs based on whether interaction is confirmed persistent
            if assoc.is_persistent:
                line_color = (0, 69, 255)    # Orange-Red for persistent interaction
                thickness = 2
                status_text = f"INTERACTION ({assoc.duration:.1f}s)"
            else:
                line_color = (0, 215, 255)   # Amber-Yellow for candidate proximity
                thickness = 1
                status_text = f"PROXIMITY ({assoc.duration:.1f}s)"

            # Draw vehicle's expanded interaction zone
            """ix1, iy1, ix2, iy2 = get_vehicle_interaction_zone(
                vehicle.bbox,
                margin_px=self.config.interaction_margin_px,
            )
            cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), line_color, 1)"""

            # Draw connecting line between person center and vehicle center
            p_center = person.center
            v_center = vehicle.center
            cv2.line(annotated, p_center, v_center, line_color, thickness, lineType=cv2.LINE_AA)

            # Draw mid-point association badge
            mid_x = (p_center[0] + v_center[0]) // 2
            mid_y = (p_center[1] + v_center[1]) // 2

            badge_text = f"{assoc.label} | {status_text}"
            (tw, th), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            # Background rectangle for badge readability
            bx1 = mid_x - tw // 2 - 4
            by1 = mid_y - th - 4
            bx2 = bx1 + tw + 8
            by2 = mid_y + 4

            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), line_color, 1)
            cv2.putText(
                annotated,
                badge_text,
                (bx1 + 4, by2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def reset(self) -> None:
        """Reset internal association state and history."""
        self._active_associations.clear()
        self._last_seen_frame.clear()
