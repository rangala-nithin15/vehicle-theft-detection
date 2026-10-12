"""Unit tests for the SQLite event database persistence module."""

import tempfile
from pathlib import Path
import pytest

from app.behavior import VehicleState
from app.database import EventDatabase
from app.risk_engine import RiskEngine, RiskLevel


@pytest.fixture
def temp_db(tmp_path: Path) -> EventDatabase:
    """Fixture providing an isolated temporary SQLite database."""
    db_file = tmp_path / "test_surveillance.db"
    return EventDatabase(db_file)


def test_database_initialization(temp_db: EventDatabase) -> None:
    """Ensure database schema is created and tables exist."""
    assert temp_db.db_path.exists()
    summary = temp_db.get_security_summary()
    assert summary["total_events"] == 0
    assert summary["total_alerts"] == 0


def test_log_and_get_behavior_events(temp_db: EventDatabase) -> None:
    """Test logging behavior events and retrieving them."""
    row_id_1 = temp_db.log_behavior_event(
        frame_idx=10,
        timestamp=0.4,
        vehicle_id=1,
        previous_state="PARKED",
        new_state="PERSON_APPROACHING",
        reason="Person #5 entered interaction zone",
    )
    assert row_id_1 > 0

    row_id_2 = temp_db.log_behavior_event(
        frame_idx=25,
        timestamp=1.0,
        vehicle_id=2,
        previous_state="PARKED",
        new_state="VEHICLE_MOVING",
        reason="Displacement exceeded threshold",
    )
    assert row_id_2 > row_id_1

    events = temp_db.get_recent_events(limit=10)
    assert len(events) == 2
    # Ordered latest first
    assert events[0]["vehicle_id"] == 2
    assert events[0]["new_state"] == "VEHICLE_MOVING"
    assert events[1]["vehicle_id"] == 1
    assert events[1]["new_state"] == "PERSON_APPROACHING"


def test_filter_events_by_vehicle_id(temp_db: EventDatabase) -> None:
    """Test retrieving events filtered by vehicle track ID."""
    temp_db.log_behavior_event(10, 0.4, 1, "PARKED", "INTERACTION", "reason1")
    temp_db.log_behavior_event(20, 0.8, 2, "PARKED", "VEHICLE_MOVING", "reason2")
    temp_db.log_behavior_event(30, 1.2, 1, "INTERACTION", "ACCESS_ATTEMPT", "reason3")

    v1_events = temp_db.get_recent_events(vehicle_id=1)
    assert len(v1_events) == 2
    assert all(e["vehicle_id"] == 1 for e in v1_events)

    v2_events = temp_db.get_recent_events(vehicle_id=2)
    assert len(v2_events) == 1
    assert v2_events[0]["vehicle_id"] == 2


def test_log_and_get_alerts(temp_db: EventDatabase) -> None:
    """Test logging theft alerts and score-based querying."""
    temp_db.log_alert(
        frame_idx=50,
        timestamp=2.0,
        vehicle_id=1,
        person_id=4,
        risk_score=50.0,
        risk_level="MEDIUM",
        reasons=["Unknown person", "Person near vehicle"],
        snapshot_path="",
    )

    temp_db.log_alert(
        frame_idx=100,
        timestamp=4.0,
        vehicle_id=1,
        person_id=4,
        risk_score=90.0,
        risk_level="HIGH",
        reasons=["Unknown person", "Access attempt", "Vehicle moving without authorization"],
        snapshot_path="data/output/snapshots/alert_v1_p4_4000.jpg",
    )

    # All alerts
    all_alerts = temp_db.get_recent_alerts(min_score=0)
    assert len(all_alerts) == 2

    # High-risk only (min_score=70)
    high_alerts = temp_db.get_recent_alerts(min_score=70)
    assert len(high_alerts) == 1
    assert high_alerts[0]["risk_score"] == 90.0
    assert high_alerts[0]["risk_level"] == "HIGH"
    assert isinstance(high_alerts[0]["reasons"], list)
    assert "Access attempt" in high_alerts[0]["reasons"]
    assert high_alerts[0]["snapshot_path"] == "data/output/snapshots/alert_v1_p4_4000.jpg"


def test_security_summary(temp_db: EventDatabase) -> None:
    """Test security summary aggregation."""
    temp_db.log_behavior_event(10, 0.4, 1, "PARKED", "INTERACTION", "reason")
    temp_db.log_behavior_event(20, 0.8, 2, "PARKED", "VEHICLE_MOVING", "reason")

    temp_db.log_alert(10, 0.4, 1, 4, 50.0, "MEDIUM", ["Unknown person"])
    temp_db.log_alert(20, 0.8, 1, 4, 80.0, "HIGH", ["Theft risk high"])

    summary = temp_db.get_security_summary()
    assert summary["total_events"] == 2
    assert summary["total_alerts"] == 2
    assert summary["high_risk_alerts"] == 1
    assert summary["medium_risk_alerts"] == 1
    assert summary["unique_vehicles_seen"] == 2


def test_clear_history(temp_db: EventDatabase) -> None:
    """Test clearing event and alert history."""
    temp_db.log_behavior_event(10, 0.4, 1, "PARKED", "INTERACTION", "reason")
    temp_db.log_alert(10, 0.4, 1, 4, 80.0, "HIGH", ["Theft alert"])

    assert len(temp_db.get_recent_events()) == 1
    assert len(temp_db.get_recent_alerts()) == 1

    temp_db.clear_history()

    assert len(temp_db.get_recent_events()) == 0
    assert len(temp_db.get_recent_alerts()) == 0


def test_risk_engine_integration_with_db(temp_db: EventDatabase, tmp_path: Path) -> None:
    """Verify that RiskEngine automatically logs triggered high-risk alerts to the database."""
    engine = RiskEngine(snapshots_dir=tmp_path / "snapshots", database=temp_db)

    # Evaluate low-risk scenario
    engine.evaluate(
        vehicle_id=1,
        vehicle_state=VehicleState.PARKED,
        person_id=None,
        timestamp=1.0,
        frame_idx=25,
    )
    assert len(temp_db.get_recent_alerts()) == 0

    # Evaluate high-risk theft sequence (unknown person + access attempt + moving)
    # 20 (unknown) + 10 (approach) + 20 (interaction) + 20 (access) + 30 (moving) = 100
    engine.evaluate(
        vehicle_id=1,
        vehicle_state=VehicleState.VEHICLE_MOVING,
        person_id=999,  # unknown person
        timestamp=2.5,
        frame_idx=62,
    )
    alerts = temp_db.get_recent_alerts()
    assert len(alerts) == 1
    assert alerts[0]["risk_score"] >= 70
    assert alerts[0]["vehicle_id"] == 1
    assert alerts[0]["person_id"] == 999
    assert alerts[0]["frame_idx"] == 62
