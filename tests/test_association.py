"""Unit tests for ParkingZone, interaction zones, and PersonVehicleAssociation (Phase 3)."""

import math
import numpy as np
import pytest

from app.association import (
    AssociationEngine,
    PersonVehicleAssociation,
    euclidean_distance,
    point_to_rect_distance,
)
from app.config import DetectionConfig
from app.tracker import TrackedObject
from app.zone import ParkingZone, get_vehicle_interaction_zone, is_point_in_rect


def test_parking_zone_point_in_polygon() -> None:
    """Test point-in-polygon containment for inside, outside, and boundary coordinates."""
    points = [
        [100, 100],
        [500, 100],
        [500, 400],
        [100, 400],
    ]
    zone = ParkingZone(points, name="Test Parking")

    # Point clearly inside
    assert zone.contains_point((250, 250)) is True
    # Point on boundary / vertex
    assert zone.contains_point((100, 100)) is True
    assert zone.contains_point((500, 250)) is True
    # Points clearly outside
    assert zone.contains_point((50, 50)) is False
    assert zone.contains_point((600, 250)) is False
    assert zone.contains_point((250, 500)) is False


def test_parking_zone_entering_and_leaving() -> None:
    """Test vehicle bounding box entering and leaving the parking zone."""
    points = [
        [100, 100],
        [500, 100],
        [500, 400],
        [100, 400],
    ]
    zone = ParkingZone(points)

    # State 1: Outside (before entering parking lot)
    box_outside = (20, 20, 80, 80)
    assert zone.contains_box(box_outside) is False

    # State 2: Inside (parked inside parking lot)
    box_inside = (200, 150, 300, 250)
    assert zone.contains_box(box_inside) is True

    # State 3: Left parking lot
    box_left = (600, 200, 750, 350)
    assert zone.contains_box(box_left) is False


def test_distance_calculations() -> None:
    """Test Euclidean point distance and point-to-rectangle shortest distance."""
    # Point to point
    dist = euclidean_distance((0, 0), (300, 400))
    assert dist == pytest.approx(500.0)

    # Point to rectangle (rect: x1=100, y1=100, x2=200, y2=200)
    rect = (100, 100, 200, 200)

    # Point inside rectangle -> distance is 0.0
    assert point_to_rect_distance((150, 150), rect) == 0.0

    # Point directly to the right by 50px
    assert point_to_rect_distance((250, 150), rect) == pytest.approx(50.0)

    # Point directly above by 40px
    assert point_to_rect_distance((150, 60), rect) == pytest.approx(40.0)

    # Diagonal distance: dx=30, dy=40 -> hypot = 50
    assert point_to_rect_distance((230, 60), rect) == pytest.approx(50.0)


def test_vehicle_interaction_zone() -> None:
    """Test calculation and containment for expanded vehicle interaction area."""
    bbox = (100, 100, 200, 200)
    margin = 40
    izone = get_vehicle_interaction_zone(bbox, margin_px=margin)

    assert izone == (60, 60, 240, 240)

    # Point outside vehicle bbox, but inside interaction zone
    pt_near = (80, 80)
    assert is_point_in_rect(pt_near, bbox) is False
    assert is_point_in_rect(pt_near, izone) is True

    # Point far outside interaction zone
    pt_far = (300, 300)
    assert is_point_in_rect(pt_far, izone) is False


def test_association_thresholds() -> None:
    """Test that nearby person produces an association while far person does not."""
    cfg = DetectionConfig(
        max_association_distance_px=100.0,
        interaction_margin_px=30,
        min_persistence_frames=3,
    )
    zone = ParkingZone([[0, 0], [1000, 0], [1000, 1000], [0, 1000]])
    engine = AssociationEngine(config=cfg, parking_zone=zone)

    # Parked Motorcycle #12
    motorcycle = TrackedObject(
        track_id=12,
        class_name="motorcycle",
        confidence=0.90,
        bbox=(200, 200, 300, 300),
    )

    # Person #1 near motorcycle (20 pixels away)
    person_near = TrackedObject(
        track_id=1,
        class_name="person",
        confidence=0.88,
        bbox=(150, 180, 180, 280),
    )

    # Person #2 far away (500 pixels away)
    person_far = TrackedObject(
        track_id=2,
        class_name="person",
        confidence=0.85,
        bbox=(700, 700, 750, 850),
    )

    assocs = engine.update([motorcycle, person_near, person_far], frame_idx=1, timestamp=0.04)

    # Only Person #1 should be associated with Motorcycle #12
    assert len(assocs) == 1
    assert assocs[0].person_id == 1
    assert assocs[0].vehicle_id == 12
    assert assocs[0].vehicle_class == "motorcycle"
    assert assocs[0].label == "Person #1 -> Motorcycle #12"


def test_persistence_logic() -> None:
    """Test that an association is candidate on frame 1, and becomes persistent only after threshold."""
    cfg = DetectionConfig(
        max_association_distance_px=100.0,
        min_persistence_frames=5,
        min_persistence_seconds=1.0,
    )
    zone = ParkingZone([[0, 0], [1000, 0], [1000, 1000], [0, 1000]])
    engine = AssociationEngine(config=cfg, parking_zone=zone)

    car = TrackedObject(track_id=3, class_name="car", confidence=0.92, bbox=(200, 200, 400, 350))
    person = TrackedObject(track_id=17, class_name="person", confidence=0.89, bbox=(150, 220, 190, 340))

    # Frame 1: Candidate, NOT persistent yet
    a1 = engine.update([car, person], frame_idx=1, timestamp=0.1)
    assert len(a1) == 1
    assert a1[0].frame_count == 1
    assert a1[0].is_persistent is False

    # Frames 2 to 4: Still not meeting 5 frames threshold
    for f in range(2, 5):
        a_mid = engine.update([car, person], frame_idx=f, timestamp=0.1 * f)
        assert a_mid[0].is_persistent is False

    # Frame 5: Meets min_persistence_frames (5) -> becomes persistent!
    a5 = engine.update([car, person], frame_idx=5, timestamp=0.5)
    assert len(a5) == 1
    assert a5[0].frame_count == 5
    assert a5[0].is_persistent is True
    assert a5[0].duration == pytest.approx(0.4, rel=1e-2)


def test_association_cooldown_cleanup() -> None:
    """Test that an association is purged after cooldown frames when person walks away."""
    cfg = DetectionConfig(
        max_association_distance_px=50.0,
        association_cooldown_frames=3,
    )
    zone = ParkingZone([[0, 0], [1000, 0], [1000, 1000], [0, 1000]])
    engine = AssociationEngine(config=cfg, parking_zone=zone)

    car = TrackedObject(track_id=5, class_name="car", confidence=0.9, bbox=(100, 100, 300, 300))
    person_near = TrackedObject(track_id=2, class_name="person", confidence=0.9, bbox=(80, 150, 95, 250))
    person_far = TrackedObject(track_id=2, class_name="person", confidence=0.9, bbox=(800, 800, 850, 950))

    # Frame 1: Associated
    a1 = engine.update([car, person_near], frame_idx=1, timestamp=0.1)
    assert len(a1) == 1
    assert (2, 5) in engine._active_associations

    # Person moves away
    engine.update([car, person_far], frame_idx=2, timestamp=0.2)
    engine.update([car, person_far], frame_idx=3, timestamp=0.3)
    engine.update([car, person_far], frame_idx=4, timestamp=0.4)
    # Frame 5: frame_idx(5) - last_seen(1) = 4 > cooldown(3) -> Purged!
    engine.update([car, person_far], frame_idx=5, timestamp=0.5)

    assert (2, 5) not in engine._active_associations
