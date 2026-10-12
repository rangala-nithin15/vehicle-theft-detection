"""Temporal behavior state machine and vehicle movement analysis module."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.association import (
    PersonVehicleAssociation,
    euclidean_distance,
    point_to_rect_distance,
)
from app.config import DetectionConfig, config as default_config
from app.tracker import TrackedObject
from app.zone import ParkingZone


class VehicleState(str, Enum):
    """Temporal lifecycle and interaction states of a monitored vehicle."""

    PARKED = "PARKED"
    PERSON_APPROACHING = "PERSON_APPROACHING"
    PERSON_NEAR = "PERSON_NEAR"
    INTERACTION = "INTERACTION"
    ACCESS_ATTEMPT = "ACCESS_ATTEMPT"
    VEHICLE_MOVING = "VEHICLE_MOVING"
    EXITING = "EXITING"


# BGR colors for each vehicle state
STATE_COLORS: Dict[VehicleState, Tuple[int, int, int]] = {
    VehicleState.PARKED: (0, 200, 0),
    VehicleState.PERSON_APPROACHING: (255, 255, 0),
    VehicleState.PERSON_NEAR: (0, 255, 255),
    VehicleState.INTERACTION: (0, 165, 255),
    VehicleState.ACCESS_ATTEMPT: (0, 69, 255),
    VehicleState.VEHICLE_MOVING: (255, 0, 255),
    VehicleState.EXITING: (0, 0, 255),
}


@dataclass
class StateTransitionEvent:
    """Represents an atomic state transition event in a vehicle's lifecycle."""

    timestamp: float
    iso_timestamp: str
    person_id: Optional[int]
    vehicle_id: int
    previous_state: VehicleState
    new_state: VehicleState
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert state transition event to serializable dictionary."""

        return {
            "timestamp": self.iso_timestamp,
            "time_seconds": round(self.timestamp, 2),
            "person_id": self.person_id,
            "vehicle_id": self.vehicle_id,
            "previous_state": self.previous_state.value,
            "new_state": self.new_state.value,
            "reason": self.reason,
        }


@dataclass
class VehicleBehaviorTracker:
    """Maintains state history and movement tracking for a single vehicle."""

    vehicle_id: int
    current_state: VehicleState = VehicleState.PARKED
    state_enter_time: float = 0.0
    state_enter_frame: int = 0

    # Last known associated person.
    # This is intentionally retained during temporary association loss.
    associated_person_id: Optional[int] = None

    # Trajectory points for displacement estimation: deque of (x, y)
    center_history: Deque[Tuple[int, int]] = None  # type: ignore

    # Movement consecutive counter
    moving_consecutive_frames: int = 0
    stationary_consecutive_frames: int = 0

    def __post_init__(self) -> None:
        if self.center_history is None:
            self.center_history = deque(maxlen=30)


class BehaviorEngine:
    """Manages temporal state transitions and movement detection for all monitored vehicles."""

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        parking_zone: Optional[ParkingZone] = None,
    ) -> None:
        """Initialize the behavior engine."""

        self.config = config or default_config
        self.parking_zone = parking_zone or ParkingZone(
            self.config.parking_zone_points
        )

        # Map vehicle_id -> VehicleBehaviorTracker
        self.vehicles: Dict[int, VehicleBehaviorTracker] = {}

        # Chronological audit log of state transition events
        self.events: List[StateTransitionEvent] = []

    def update(
        self,
        tracked_objects: List[TrackedObject],
        associations: List[PersonVehicleAssociation],
        frame_idx: int = 0,
        timestamp: float = 0.0,
    ) -> Dict[int, VehicleState]:
        """Update temporal behavior states for all vehicles in the current frame."""

        # Separate persons and vehicles
        persons = [
            obj
            for obj in tracked_objects
            if obj.class_name == "person"
        ]

        vehicles = [
            obj
            for obj in tracked_objects
            if obj.class_name
            in {"car", "motorcycle", "bicycle", "truck", "bus"}
        ]

        # Map active associations by vehicle_id
        assoc_by_vehicle: Dict[int, PersonVehicleAssociation] = {}

        for assoc in associations:
            assoc_by_vehicle[assoc.vehicle_id] = assoc

        current_states: Dict[int, VehicleState] = {}

        for vehicle in vehicles:
            v_id = vehicle.track_id

            # Initialize tracker state if this is the first time
            # observing this vehicle
            if v_id not in self.vehicles:
                self.vehicles[v_id] = VehicleBehaviorTracker(
                    vehicle_id=v_id,
                    current_state=VehicleState.PARKED,
                    state_enter_time=timestamp,
                    state_enter_frame=frame_idx,
                )

            v_tracker = self.vehicles[v_id]

            # Store vehicle trajectory
            v_tracker.center_history.append(vehicle.center)

            # 1. Evaluate vehicle physical movement
            is_moving, displacement = self._check_movement(v_tracker)

            # 2. Check parking zone containment
            inside_parking_zone = self.parking_zone.contains_box(
                vehicle.bbox,
                check_bottom_center=True,
            )

            # 3. Check person proximity & association
            active_assoc = assoc_by_vehicle.get(v_id)

            min_person_dist, closest_person_id = self._find_closest_person(
                vehicle,
                persons,
            )

            # 4. Determine next state
            prev_state = v_tracker.current_state

            next_state, person_id, reason = self._compute_next_state(
                v_tracker=v_tracker,
                vehicle=vehicle,
                active_assoc=active_assoc,
                is_moving=is_moving,
                displacement=displacement,
                inside_parking_zone=inside_parking_zone,
                min_person_dist=min_person_dist,
                closest_person_id=closest_person_id,
                timestamp=timestamp,
                frame_idx=frame_idx,
            )

            # 5. Handle state transition if changed
            if next_state != prev_state:
                self._transition_to(
                    v_tracker=v_tracker,
                    new_state=next_state,
                    person_id=person_id,
                    reason=reason,
                    timestamp=timestamp,
                    frame_idx=frame_idx,
                )
            else:
                # IMPORTANT:
                # Even when the state does not change, retain the latest
                # meaningful person association.
                if person_id is not None:
                    v_tracker.associated_person_id = person_id

            current_states[v_id] = v_tracker.current_state

        return current_states

    def _check_movement(
        self,
        v_tracker: VehicleBehaviorTracker,
    ) -> Tuple[bool, float]:
        """Check whether the vehicle is genuinely moving based on trajectory displacement."""

        history = v_tracker.center_history

        window = min(
            len(history),
            self.config.movement_window_frames,
        )

        if window < 2:
            return False, 0.0

        # Calculate displacement between oldest and newest
        # position in the window
        pt_old = list(history)[-window]
        pt_new = list(history)[-1]

        displacement = euclidean_distance(
            pt_old,
            pt_new,
        )

        if displacement > self.config.movement_threshold_px:
            v_tracker.moving_consecutive_frames += 1
            v_tracker.stationary_consecutive_frames = 0

        else:
            v_tracker.stationary_consecutive_frames += 1

            if (
                v_tracker.stationary_consecutive_frames
                >= self.config.movement_window_frames
            ):
                v_tracker.moving_consecutive_frames = 0

        is_moving = (
            v_tracker.moving_consecutive_frames
            >= self.config.movement_consecutive_frames
        )

        return is_moving, displacement

    def _find_closest_person(
        self,
        vehicle: TrackedObject,
        persons: List[TrackedObject],
    ) -> Tuple[float, Optional[int]]:
        """Find the distance to the closest person."""

        if not persons:
            return float("inf"), None

        min_dist = float("inf")
        closest_id = None

        for p in persons:
            # Person bottom-center represents feet position
            # on the ground plane.
            p_feet = (
                p.center[0],
                p.y2,
            )

            dist = point_to_rect_distance(
                p_feet,
                vehicle.bbox,
            )

            if dist < min_dist:
                min_dist = dist
                closest_id = p.track_id

        return min_dist, closest_id

    def _compute_next_state(
        self,
        v_tracker: VehicleBehaviorTracker,
        vehicle: TrackedObject,
        active_assoc: Optional[PersonVehicleAssociation],
        is_moving: bool,
        displacement: float,
        inside_parking_zone: bool,
        min_person_dist: float,
        closest_person_id: Optional[int],
        timestamp: float,
        frame_idx: int,
    ) -> Tuple[VehicleState, Optional[int], str]:
        """Compute the next state using rule-based temporal logic."""

        current = v_tracker.current_state

        time_in_state = max(
            0.0,
            timestamp - v_tracker.state_enter_time,
        )

        # ============================================================
        # Rule A: Moving vehicles
        # ============================================================

        if is_moving:

            if not inside_parking_zone:
                return (
                    VehicleState.EXITING,
                    v_tracker.associated_person_id,
                    (
                        f"Vehicle moving outside parking zone "
                        f"({displacement:.1f}px)"
                    ),
                )

            return (
                VehicleState.VEHICLE_MOVING,
                v_tracker.associated_person_id,
                (
                    f"Vehicle in motion "
                    f"({displacement:.1f}px displacement)"
                ),
            )

        # ============================================================
        # Rule B: Vehicle stopped after moving
        # ============================================================

        if (
            current in (
                VehicleState.VEHICLE_MOVING,
                VehicleState.EXITING,
            )
            and not is_moving
        ):

            if time_in_state >= self.config.stationary_duration_sec:

                if inside_parking_zone:
                    return (
                        VehicleState.PARKED,
                        None,
                        "Vehicle became stationary inside parking zone",
                    )

        # ============================================================
        # Rule C: Person association and interaction progression
        # ============================================================

        if active_assoc is not None:

            # We have a confirmed current association.
            p_id = active_assoc.person_id

            # Store the latest associated person.
            v_tracker.associated_person_id = p_id

            # --------------------------------------------------------
            # 1. ACCESS_ATTEMPT
            # --------------------------------------------------------

            if (
                active_assoc.duration
                >= self.config.access_attempt_duration_sec
            ):
                return (
                    VehicleState.ACCESS_ATTEMPT,
                    p_id,
                    (
                        f"Prolonged interaction "
                        f"({active_assoc.duration:.1f}s) "
                        f"indicates access attempt"
                    ),
                )

            # --------------------------------------------------------
            # 2. INTERACTION
            # --------------------------------------------------------

            if (
                active_assoc.is_persistent
                and active_assoc.duration
                >= self.config.interaction_duration_sec
            ):
                return (
                    VehicleState.INTERACTION,
                    p_id,
                    (
                        f"Persistent interaction confirmed "
                        f"({active_assoc.duration:.1f}s)"
                    ),
                )

            # --------------------------------------------------------
            # 3. PERSON_NEAR
            # --------------------------------------------------------

            return (
                VehicleState.PERSON_NEAR,
                p_id,
                (
                    f"Person #{p_id} within interaction "
                    f"proximity ({active_assoc.distance:.1f}px)"
                ),
            )

        # ============================================================
        # Rule D: No active association
        #
        # Check whether a person is still physically close.
        # ============================================================

        if closest_person_id is not None:

            if min_person_dist <= self.config.near_distance_px:

                # Update last known association.
                v_tracker.associated_person_id = closest_person_id

                return (
                    VehicleState.PERSON_NEAR,
                    closest_person_id,
                    (
                        f"Person #{closest_person_id} near vehicle "
                        f"({min_person_dist:.1f}px)"
                    ),
                )

            if min_person_dist <= self.config.approaching_distance_px:

                # Update last known association.
                v_tracker.associated_person_id = closest_person_id

                return (
                    VehicleState.PERSON_APPROACHING,
                    closest_person_id,
                    (
                        f"Person #{closest_person_id} approaching "
                        f"({min_person_dist:.1f}px)"
                    ),
                )

        # ============================================================
        # Rule D2: Temporary association loss
        #
        # IMPORTANT:
        # Do NOT immediately erase the last associated person.
        #
        # This is important for CCTV because a person may temporarily
        # disappear from detection while the vehicle is still moving.
        # ============================================================

        if v_tracker.associated_person_id is not None:

            return (
                current,
                v_tracker.associated_person_id,
                (
                    "Temporary loss of person detection; "
                    "retaining last known association"
                ),
            )

        # ============================================================
        # Rule E: No persons nearby and no previous association
        # ============================================================

        v_tracker.associated_person_id = None

        return (
            VehicleState.PARKED,
            None,
            "Vehicle parked and undisturbed",
        )

    def _transition_to(
        self,
        v_tracker: VehicleBehaviorTracker,
        new_state: VehicleState,
        person_id: Optional[int],
        reason: str,
        timestamp: float,
        frame_idx: int,
    ) -> None:
        """Execute and record state transition."""

        prev_state = v_tracker.current_state

        # Create structured event record
        iso_time = datetime.now(timezone.utc).isoformat()

        event = StateTransitionEvent(
            timestamp=timestamp,
            iso_timestamp=iso_time,
            person_id=person_id,
            vehicle_id=v_tracker.vehicle_id,
            previous_state=prev_state,
            new_state=new_state,
            reason=reason,
        )

        self.events.append(event)

        # Update tracker
        v_tracker.current_state = new_state
        v_tracker.state_enter_time = timestamp
        v_tracker.state_enter_frame = frame_idx
        v_tracker.associated_person_id = person_id

    def draw_vehicle_states(
        self,
        frame: np.ndarray,
        vehicle_states: Dict[int, VehicleState],
        tracked_objects: List[TrackedObject],
    ) -> np.ndarray:
        """Annotate vehicle bounding boxes with current behavior state badge."""

        annotated = frame.copy()

        vehicles = [
            obj
            for obj in tracked_objects
            if obj.class_name
            in {"car", "motorcycle", "bicycle", "truck", "bus"}
        ]

        for vehicle in vehicles:

            v_id = vehicle.track_id

            state = vehicle_states.get(
                v_id,
                VehicleState.PARKED,
            )

            state_color = STATE_COLORS.get(
                state,
                (0, 200, 0),
            )

            x1, y1, x2, y2 = vehicle.bbox

            # State badge text
            v_tracker = self.vehicles.get(v_id)

            if (
                v_tracker
                and v_tracker.associated_person_id is not None
            ):
                state_text = (
                    f"State: {state.value} "
                    f"(P#{v_tracker.associated_person_id})"
                )
            else:
                state_text = f"State: {state.value}"

            # Badge dimensions
            (tw, th), baseline = cv2.getTextSize(
                state_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
            )

            # Draw state badge below bbox
            bx1 = x1
            by1 = y2 + 4
            bx2 = bx1 + tw + 8
            by2 = by1 + th + 6

            # Avoid falling off the bottom of the frame
            if by2 >= frame.shape[0]:
                by1 = max(
                    0,
                    y1 - th - 26,
                )

                by2 = by1 + th + 6

            cv2.rectangle(
                annotated,
                (bx1, by1),
                (bx2, by2),
                (20, 20, 20),
                -1,
            )

            cv2.rectangle(
                annotated,
                (bx1, by1),
                (bx2, by2),
                state_color,
                1,
            )

            cv2.putText(
                annotated,
                state_text,
                (bx1 + 4, by2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                state_color,
                1,
                cv2.LINE_AA,
            )

        return annotated

    def save_events_json(
        self,
        output_path: Path,
    ) -> None:
        """Save collected state transition events to a formatted JSON file."""

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = [
            e.to_dict()
            for e in self.events
        ]

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                indent=2,
            )

    def reset(self) -> None:
        """Reset internal vehicle states and event history."""

        self.vehicles.clear()
        self.events.clear()