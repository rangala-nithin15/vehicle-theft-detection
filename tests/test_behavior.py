"""Unit tests for temporal vehicle access behavior state machine (Phase 4)."""

import json
from pathlib import Path
import pytest

from app.association import PersonVehicleAssociation
from app.behavior import BehaviorEngine, StateTransitionEvent, VehicleState
from app.config import DetectionConfig
from app.tracker import TrackedObject
from app.zone import ParkingZone


def test_movement_thresholding_and_jitter_rejection() -> None:
    """Test that small bounding-box jitter does not trigger movement, but genuine displacement does."""
    cfg = DetectionConfig(
        movement_threshold_px=18.0,
        movement_window_frames=5,
        movement_consecutive_frames=3,
    )
    zone = ParkingZone([[0, 0], [1000, 0], [1000, 1000], [0, 1000]])
    engine = BehaviorEngine(config=cfg, parking_zone=zone)

    v_id = 1
    # 1. Simulate camera jitter: fluctuating by 1-2 pixels
    jitter_positions = [(200, 200), (201, 202), (200, 201), (202, 200), (201, 201), (200, 200)]
    for i, (cx, cy) in enumerate(jitter_positions):
        veh = TrackedObject(
            track_id=v_id,
            class_name="car",
            confidence=0.9,
            bbox=(cx - 50, cy - 30, cx + 50, cy + 30),
        )
        states = engine.update([veh], associations=[], frame_idx=i + 1, timestamp=0.1 * (i + 1))
        # Must remain PARKED
        assert states[v_id] == VehicleState.PARKED

    # 2. Simulate genuine vehicle motion (large displacements)
    moving_positions = [(230, 230), (260, 260), (300, 300), (350, 350), (400, 400)]
    latest_state = VehicleState.PARKED
    for i, (cx, cy) in enumerate(moving_positions, start=len(jitter_positions)):
        veh = TrackedObject(
            track_id=v_id,
            class_name="car",
            confidence=0.9,
            bbox=(cx - 50, cy - 30, cx + 50, cy + 30),
        )
        states = engine.update([veh], associations=[], frame_idx=i + 1, timestamp=0.1 * (i + 1))
        latest_state = states[v_id]

    # Must transition to VEHICLE_MOVING
    assert latest_state == VehicleState.VEHICLE_MOVING


def test_parked_to_person_near() -> None:
    """Test transition from PARKED to PERSON_NEAR when a person enters proximity."""
    cfg = DetectionConfig(near_distance_px=100.0)
    zone = ParkingZone([[0, 0], [1000, 0], [1000, 1000], [0, 1000]])
    engine = BehaviorEngine(config=cfg, parking_zone=zone)

    # Frame 1: Vehicle alone
    car = TrackedObject(track_id=10, class_name="car", confidence=0.9, bbox=(200, 200, 400, 350))
    s1 = engine.update([car], associations=[], frame_idx=1, timestamp=0.1)
    assert s1[10] == VehicleState.PARKED

    # Frame 2: Person enters near vehicle (40px away)
    person = TrackedObject(track_id=5, class_name="person", confidence=0.88, bbox=(120, 210, 160, 330))
    assoc = PersonVehicleAssociation(
        person_id=5,
        vehicle_id=10,
        vehicle_class="car",
        distance=40.0,
        center_distance=150.0,
        confidence=0.8,
        first_seen=0.2,
        last_seen=0.2,
        duration=0.0,
        frame_count=1,
        is_persistent=False,
        vehicle_in_parking_zone=True,
        in_interaction_zone=True,
    )

    s2 = engine.update([car, person], associations=[assoc], frame_idx=2, timestamp=0.2)
    assert s2[10] == VehicleState.PERSON_NEAR

    # Verify event logged
    assert len(engine.events) == 1
    ev = engine.events[0]
    assert ev.previous_state == VehicleState.PARKED
    assert ev.new_state == VehicleState.PERSON_NEAR
    assert ev.person_id == 5
    assert ev.vehicle_id == 10


def test_person_near_to_interaction() -> None:
    """Test transition from PERSON_NEAR to INTERACTION when interaction is persistent."""
    cfg = DetectionConfig(interaction_duration_sec=1.0)
    zone = ParkingZone([[0, 0], [1000, 0], [1000, 1000], [0, 1000]])
    engine = BehaviorEngine(config=cfg, parking_zone=zone)

    car = TrackedObject(track_id=10, class_name="car", confidence=0.9, bbox=(200, 200, 400, 350))
    person = TrackedObject(track_id=5, class_name="person", confidence=0.88, bbox=(150, 210, 190, 330))

    # Candidate association (duration 0.2s < 1.0s) -> PERSON_NEAR
    assoc_short = PersonVehicleAssociation(
        person_id=5,
        vehicle_id=10,
        vehicle_class="car",
        distance=20.0,
        center_distance=100.0,
        confidence=0.85,
        first_seen=0.1,
        last_seen=0.3,
        duration=0.2,
        frame_count=3,
        is_persistent=False,
        vehicle_in_parking_zone=True,
        in_interaction_zone=True,
    )
    s1 = engine.update([car, person], associations=[assoc_short], frame_idx=1, timestamp=0.3)
    assert s1[10] == VehicleState.PERSON_NEAR

    # Persistent association with duration >= 1.0s -> INTERACTION
    assoc_persistent = PersonVehicleAssociation(
        person_id=5,
        vehicle_id=10,
        vehicle_class="car",
        distance=20.0,
        center_distance=100.0,
        confidence=0.92,
        first_seen=0.1,
        last_seen=1.5,
        duration=1.4,
        frame_count=15,
        is_persistent=True,
        vehicle_in_parking_zone=True,
        in_interaction_zone=True,
    )
    s2 = engine.update([car, person], associations=[assoc_persistent], frame_idx=15, timestamp=1.5)
    assert s2[10] == VehicleState.INTERACTION


def test_interaction_to_access_attempt() -> None:
    """Test transition from INTERACTION to ACCESS_ATTEMPT when duration is prolonged."""
    cfg = DetectionConfig(
        interaction_duration_sec=1.0,
        access_attempt_duration_sec=2.5,
    )
    zone = ParkingZone([[0, 0], [1000, 0], [1000, 1000], [0, 1000]])
    engine = BehaviorEngine(config=cfg, parking_zone=zone)

    motorcycle = TrackedObject(track_id=7, class_name="motorcycle", confidence=0.92, bbox=(200, 200, 300, 300))
    person = TrackedObject(track_id=3, class_name="person", confidence=0.90, bbox=(170, 200, 210, 300))

    # Prolonged interaction of 3.2s >= 2.5s -> ACCESS_ATTEMPT
    assoc_prolonged = PersonVehicleAssociation(
        person_id=3,
        vehicle_id=7,
        vehicle_class="motorcycle",
        distance=10.0,
        center_distance=60.0,
        confidence=0.95,
        first_seen=0.5,
        last_seen=3.7,
        duration=3.2,
        frame_count=35,
        is_persistent=True,
        vehicle_in_parking_zone=True,
        in_interaction_zone=True,
    )
    s = engine.update([motorcycle, person], associations=[assoc_prolonged], frame_idx=35, timestamp=3.7)
    assert s[7] == VehicleState.ACCESS_ATTEMPT


def test_interaction_to_moving() -> None:
    """Test transition from INTERACTION to VEHICLE_MOVING when vehicle starts moving."""
    cfg = DetectionConfig(
        movement_threshold_px=15.0,
        movement_window_frames=3,
        movement_consecutive_frames=2,
    )
    zone = ParkingZone([[0, 0], [1000, 0], [1000, 1000], [0, 1000]])
    engine = BehaviorEngine(config=cfg, parking_zone=zone)

    # Frame 1: Persistent interaction
    v1 = TrackedObject(track_id=8, class_name="car", confidence=0.9, bbox=(200, 200, 350, 300))
    p1 = TrackedObject(track_id=4, class_name="person", confidence=0.9, bbox=(180, 200, 220, 300))
    assoc = PersonVehicleAssociation(
        person_id=4,
        vehicle_id=8,
        vehicle_class="car",
        distance=15.0,
        center_distance=50.0,
        confidence=0.9,
        first_seen=0.1,
        last_seen=1.2,
        duration=1.1,
        frame_count=12,
        is_persistent=True,
        vehicle_in_parking_zone=True,
        in_interaction_zone=True,
    )
    engine.update([v1, p1], associations=[assoc], frame_idx=1, timestamp=1.2)

    # Frame 2 & 3: Vehicle starts moving rapidly
    v2 = TrackedObject(track_id=8, class_name="car", confidence=0.9, bbox=(240, 200, 390, 300))
    engine.update([v2], associations=[], frame_idx=2, timestamp=1.3)

    v3 = TrackedObject(track_id=8, class_name="car", confidence=0.9, bbox=(290, 200, 440, 300))
    s3 = engine.update([v3], associations=[], frame_idx=3, timestamp=1.4)

    assert s3[8] == VehicleState.VEHICLE_MOVING


def test_moving_to_exiting() -> None:
    """Test transition from VEHICLE_MOVING to EXITING when crossing outside parking zone."""
    cfg = DetectionConfig(
        movement_threshold_px=10.0,
        movement_window_frames=3,
        movement_consecutive_frames=2,
    )
    # Monitored parking zone up to x=500
    zone = ParkingZone([[0, 0], [500, 0], [500, 500], [0, 500]])
    engine = BehaviorEngine(config=cfg, parking_zone=zone)

    # Moving inside parking zone
    v1 = TrackedObject(track_id=2, class_name="car", confidence=0.9, bbox=(200, 200, 300, 280))
    v2 = TrackedObject(track_id=2, class_name="car", confidence=0.9, bbox=(250, 200, 350, 280))
    v3 = TrackedObject(track_id=2, class_name="car", confidence=0.9, bbox=(300, 200, 400, 280))
    engine.update([v1], associations=[], frame_idx=1, timestamp=0.1)
    engine.update([v2], associations=[], frame_idx=2, timestamp=0.2)
    s3 = engine.update([v3], associations=[], frame_idx=3, timestamp=0.3)
    assert s3[2] == VehicleState.VEHICLE_MOVING

    # Continues moving outside parking zone boundary (x > 500)
    v4 = TrackedObject(track_id=2, class_name="car", confidence=0.9, bbox=(550, 200, 650, 280))
    s4 = engine.update([v4], associations=[], frame_idx=4, timestamp=0.4)

    assert s4[2] == VehicleState.EXITING


def test_invalid_transitions_rejected() -> None:
    """Test that a stationary vehicle without a person does not spontaneously jump to active states."""
    zone = ParkingZone([[0, 0], [1000, 0], [1000, 1000], [0, 1000]])
    engine = BehaviorEngine(parking_zone=zone)

    # Stationary car sitting undisturbed
    car = TrackedObject(track_id=1, class_name="car", confidence=0.9, bbox=(200, 200, 400, 350))
    for f in range(1, 10):
        states = engine.update([car], associations=[], frame_idx=f, timestamp=0.1 * f)
        assert states[1] == VehicleState.PARKED

    # Ensure no illegitimate state transition events were fired
    assert len(engine.events) == 0


def test_structured_event_logging(tmp_path: Path) -> None:
    """Test that state transition events follow the JSON schema and save properly to disk."""
    zone = ParkingZone([[0, 0], [1000, 0], [1000, 1000], [0, 1000]])
    engine = BehaviorEngine(parking_zone=zone)

    car = TrackedObject(track_id=12, class_name="motorcycle", confidence=0.9, bbox=(200, 200, 300, 300))
    person = TrackedObject(track_id=17, class_name="person", confidence=0.85, bbox=(180, 200, 220, 280))
    assoc = PersonVehicleAssociation(
        person_id=17,
        vehicle_id=12,
        vehicle_class="motorcycle",
        distance=15.0,
        center_distance=50.0,
        confidence=0.9,
        first_seen=0.1,
        last_seen=1.5,
        duration=1.4,
        frame_count=15,
        is_persistent=True,
        vehicle_in_parking_zone=True,
        in_interaction_zone=True,
    )

    # Trigger transition: PARKED -> INTERACTION
    engine.update([car, person], associations=[assoc], frame_idx=1, timestamp=1.5)

    assert len(engine.events) > 0
    ev = engine.events[0]
    data = ev.to_dict()

    assert "timestamp" in data
    assert data["person_id"] == 17
    assert data["vehicle_id"] == 12
    assert data["previous_state"] == "PARKED"
    assert data["new_state"] == "INTERACTION"

    # Save to JSON file
    out_file = tmp_path / "events.json"
    engine.save_events_json(out_file)

    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded) == 1
    assert loaded[0]["vehicle_id"] == 12
    assert loaded[0]["new_state"] == "INTERACTION"
