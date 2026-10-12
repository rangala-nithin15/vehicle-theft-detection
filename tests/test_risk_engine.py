"""Unit tests for AuthorizationStore, RiskEngine, and theft risk scoring (Phase 5)."""

import json
from pathlib import Path
import numpy as np
import pytest

from app.authorization import AuthorizationStore
from app.behavior import VehicleState
from app.config import DetectionConfig
from app.risk_engine import RiskAssessment, RiskEngine, RiskLevel


def test_authorization_store() -> None:
    """Test authorization matching, missing rules, and wildcard behavior."""
    rules = {
        "vehicles": {
            "12": {
                "authorized_persons": [1, 2, "admin_user"]
            },
            "100": {
                "authorized_persons": [5]
            }
        }
    }
    store = AuthorizationStore(rules)

    # Person 1 is authorized for vehicle 12
    assert store.is_authorized(1, 12) is True
    assert store.is_authorized("1", "12") is True
    assert store.is_authorized(2, 12) is True
    assert store.is_authorized("admin_user", 12) is True

    # Person 3 is NOT authorized for vehicle 12
    assert store.is_authorized(3, 12) is False

    # Vehicle 999 is unregistered -> unauthorized
    assert store.is_authorized(1, 999) is False

    # None person
    assert store.is_authorized(None, 12) is False


def test_scenario_1_known_person_parked_vehicle() -> None:
    """Scenario 1: Known/authorized person + parked vehicle -> Low risk."""
    store = AuthorizationStore({"vehicles": {"12": {"authorized_persons": [1]}}})
    engine = RiskEngine(auth_store=store)

    assessment = engine.evaluate(
        vehicle_id=12,
        vehicle_state=VehicleState.PARKED,
        person_id=1,
        timestamp=1.0,
    )

    assert assessment.is_authorized is True
    assert assessment.risk_score == 0
    assert assessment.risk_level == RiskLevel.LOW
    assert assessment.alert_triggered is False
    assert "Unknown person" not in assessment.reasons


def test_scenario_2_unknown_person_near_vehicle() -> None:
    """Scenario 2: Unknown person near vehicle -> Suspicious / Medium-Low."""
    store = AuthorizationStore({"vehicles": {"12": {"authorized_persons": [1]}}})
    engine = RiskEngine(auth_store=store)

    assessment = engine.evaluate(
        vehicle_id=12,
        vehicle_state=VehicleState.PERSON_NEAR,
        person_id=99,  # Unknown person
        timestamp=2.0,
    )

    # Unknown person (+20) + person near (+10) = 30
    assert assessment.is_authorized is False
    assert assessment.risk_score == 30
    assert assessment.risk_level == RiskLevel.LOW  # Below 40 threshold
    assert assessment.alert_triggered is False
    assert "Unknown person" in assessment.reasons
    assert "Person near vehicle" in assessment.reasons


def test_scenario_3_unknown_person_interaction() -> None:
    """Scenario 3: Unknown person + interaction -> Higher risk (Medium)."""
    store = AuthorizationStore({"vehicles": {"12": {"authorized_persons": [1]}}})
    engine = RiskEngine(auth_store=store)

    assessment = engine.evaluate(
        vehicle_id=12,
        vehicle_state=VehicleState.INTERACTION,
        person_id=99,  # Unknown person
        timestamp=3.0,
    )

    # Unknown person (+20) + approach (+10) + interaction (+20) = 50
    assert assessment.risk_score == 50
    assert assessment.risk_level == RiskLevel.MEDIUM
    assert assessment.alert_triggered is False
    assert "Persistent interaction detected" in assessment.reasons


def test_scenario_4_unknown_person_interaction_and_movement() -> None:
    """Scenario 4: Unknown person + interaction + movement -> High risk (Potential Theft)."""
    store = AuthorizationStore({"vehicles": {"12": {"authorized_persons": [1]}}})
    engine = RiskEngine(auth_store=store)

    assessment = engine.evaluate(
        vehicle_id=12,
        vehicle_state=VehicleState.VEHICLE_MOVING,
        person_id=99,  # Unknown person
        timestamp=4.0,
    )

    # Unknown (+20) + movement (+30) + in motion during interaction (+20) = 70
    assert assessment.risk_score >= 70
    assert assessment.risk_level == RiskLevel.HIGH
    assert assessment.alert_triggered is True
    assert assessment.alert_label == "POTENTIAL VEHICLE THEFT"
    assert "Unknown person" in assessment.reasons
    assert "Vehicle movement detected" in assessment.reasons


def test_scenario_5_unknown_person_movement_and_exit() -> None:
    """Scenario 5: Unknown person + movement + exit -> Very high risk."""
    store = AuthorizationStore({"vehicles": {"12": {"authorized_persons": [1]}}})
    engine = RiskEngine(auth_store=store)

    assessment = engine.evaluate(
        vehicle_id=12,
        vehicle_state=VehicleState.EXITING,
        person_id=99,  # Unknown person
        timestamp=5.0,
    )

    # Unknown (+20) + movement (+30) + exit (+40) = 90
    assert assessment.risk_score >= 90
    assert assessment.risk_level == RiskLevel.HIGH
    assert assessment.alert_triggered is True
    assert "Exit boundary crossed" in assessment.reasons
    assert "Unknown person" in assessment.reasons


def test_explainable_output_schema() -> None:
    """Test that RiskAssessment serialization matches the exact required format."""
    store = AuthorizationStore({"vehicles": {"12": {"authorized_persons": [1]}}})
    engine = RiskEngine(auth_store=store)

    assessment = engine.evaluate(
        vehicle_id=12,
        vehicle_state=VehicleState.EXITING,
        person_id=17,
        timestamp=4.5,
    )

    data = assessment.to_dict()
    assert data["vehicle_id"] == "12"
    assert data["person_id"] == "17"
    assert isinstance(data["risk_score"], int)
    assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert isinstance(data["reasons"], list)
    assert len(data["reasons"]) > 0


def test_alert_snapshot_capture(tmp_path: Path) -> None:
    """Test that a high-risk event captures and saves an annotated evidence snapshot."""
    store = AuthorizationStore()
    snapshots_dir = tmp_path / "snapshots"
    engine = RiskEngine(auth_store=store, snapshots_dir=snapshots_dir)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    vehicle_bbox = (200, 200, 400, 350)

    # Trigger HIGH risk (EXITING with unknown person -> 90 score)
    assessment = engine.evaluate(
        vehicle_id=9,
        vehicle_state=VehicleState.EXITING,
        person_id=88,
        timestamp=2.5,
        frame=dummy_frame,
        vehicle_bbox=vehicle_bbox,
    )

    assert assessment.alert_triggered is True
    assert assessment.snapshot_path is not None
    assert Path(assessment.snapshot_path).exists()
    assert len(engine.alerts) == 1
