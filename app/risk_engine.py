"""Vehicle theft risk scoring engine combining authorization, behavior, and spatial factors."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.authorization import AuthorizationStore
from app.behavior import VehicleState
from app.config import DetectionConfig, config as default_config
from app.tracker import TrackedObject


class RiskLevel(str, Enum):
    """Categorical risk severity level."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class RiskAssessment:
    """Detailed explainable risk evaluation for a vehicle in a specific video frame."""

    vehicle_id: int
    person_id: Optional[int]
    risk_score: int
    risk_level: RiskLevel
    reasons: List[str]
    timestamp: float
    iso_timestamp: str
    vehicle_state: VehicleState
    is_authorized: bool
    alert_triggered: bool = False
    alert_label: str = "POTENTIAL VEHICLE THEFT"
    snapshot_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert risk assessment to dictionary."""

        return {
            "vehicle_id": str(self.vehicle_id),
            "person_id": (
                str(self.person_id)
                if self.person_id is not None
                else None
            ),
            "risk_score": self.risk_score,
            "risk_level": self.risk_level.value,
            "reasons": list(self.reasons),
            "vehicle_state": self.vehicle_state.value,
            "is_authorized": self.is_authorized,
            "alert_triggered": self.alert_triggered,
            "alert_label": (
                self.alert_label
                if self.alert_triggered
                else None
            ),
            "timestamp": self.iso_timestamp,
            "time_seconds": round(self.timestamp, 2),
            "snapshot_path": self.snapshot_path,
        }


class RiskEngine:
    """Transparent rule-based risk evaluation engine."""

    def __init__(
        self,
        config: Optional[DetectionConfig] = None,
        auth_store: Optional[AuthorizationStore] = None,
        snapshots_dir: Optional[Path] = None,
        database: Optional[Any] = None,
    ) -> None:

        self.config = config or default_config
        self.auth_store = auth_store or AuthorizationStore()

        self.snapshots_dir = (
            snapshots_dir
            or (self.config.output_dir / "snapshots")
        )

        self.snapshots_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.database = database

        # ============================================================
        # RISK SCORING WEIGHTS
        # ============================================================

        self.weight_unknown_person = 20
        self.weight_approach = 10
        self.weight_interaction = 20
        self.weight_access_attempt = 20
        self.weight_movement = 30
        self.weight_exit = 40

        # ============================================================
        # RISK THRESHOLDS
        # ============================================================

        self.medium_risk_threshold = 40
        self.high_risk_threshold = 70

        # ============================================================
        # ALERT STORAGE
        # ============================================================

        self.alerts: List[RiskAssessment] = []

        # ============================================================
        # SNAPSHOT COOLDOWN
        # ============================================================

        self._last_snapshot_time: Dict[int, float] = {}
        self.snapshot_cooldown_sec = 2.0

        # ============================================================
        # DUPLICATE ALERT PREVENTION
        # ============================================================

        self._high_alert_active: Dict[int, bool] = {}

    def evaluate(
        self,
        vehicle_id: int,
        vehicle_state: VehicleState,
        person_id: Optional[int],
        timestamp: float,
        frame: Optional[np.ndarray] = None,
        vehicle_bbox: Optional[Tuple[int, int, int, int]] = None,
        frame_idx: int = 0,
    ) -> RiskAssessment:
        """Calculate explainable theft risk score and assess risk level."""

        score = 0
        reasons: List[str] = []

        # ============================================================
        # 1. AUTHORIZATION CHECK
        # ============================================================

        is_authorized = False

        if person_id is not None:

            is_authorized = self.auth_store.is_authorized(
                person_id,
                vehicle_id,
            )

            if not is_authorized:

                score += self.weight_unknown_person

                reasons.append(
                    "Unknown or unauthorized person"
                )

            else:

                reasons.append(
                    "Authorized person detected"
                )

        else:

            reasons.append(
                "No associated person detected"
            )

        # ============================================================
        # 2. BEHAVIOR STATE SIGNALS
        # ============================================================

        if vehicle_state == VehicleState.PERSON_APPROACHING:

            score += self.weight_approach

            reasons.append(
                "Person approaching vehicle"
            )

        elif vehicle_state == VehicleState.PERSON_NEAR:

            score += self.weight_approach

            reasons.append(
                "Person near vehicle"
            )

        elif vehicle_state == VehicleState.INTERACTION:

            score += self.weight_approach
            score += self.weight_interaction

            reasons.append(
                "Persistent interaction detected"
            )

        elif vehicle_state == VehicleState.ACCESS_ATTEMPT:

            score += self.weight_approach
            score += self.weight_interaction
            score += self.weight_access_attempt

            reasons.append(
                "Vehicle access attempt detected"
            )

        elif vehicle_state == VehicleState.VEHICLE_MOVING:

            score += self.weight_movement

            reasons.append(
                "Vehicle movement detected"
            )

            if person_id is not None:

                score += self.weight_interaction

                reasons.append(
                    "Vehicle in motion during person interaction"
                )

        elif vehicle_state == VehicleState.EXITING:

            # --------------------------------------------------------
            # IMPORTANT:
            #
            # Vehicle movement + exit by itself is NOT enough to
            # declare vehicle theft.
            #
            # A vehicle may legitimately leave the parking area.
            # Therefore:
            #
            #   No person       -> MEDIUM at most
            #   Authorized      -> LOW/MEDIUM
            #   Unauthorized    -> HIGH
            # --------------------------------------------------------

            score += self.weight_movement
            score += self.weight_exit

            reasons.append(
                "Vehicle movement detected"
            )

            reasons.append(
                "Exit boundary crossed"
            )

            # No person detected:
            # Do not allow exit alone to become HIGH.
            if person_id is None:

                score = min(
                    score,
                    60,
                )

                reasons.append(
                    "No person association; theft not confirmed"
                )

            # Authorized person:
            # Reduce the risk because the exit has an authorized
            # person associated with the vehicle.
            elif is_authorized:

                score = max(
                    0,
                    score - 30,
                )

                reasons.append(
                    "Vehicle exiting with authorized person"
                )

            # Unauthorized person:
            # Keep the high-risk combination.
            else:

                reasons.append(
                    "Vehicle exiting with unauthorized person"
                )

        # ============================================================
        # 3. NORMALIZE RISK SCORE
        # ============================================================

        final_score = min(
            100,
            max(
                0,
                score,
            ),
        )

        # ============================================================
        # 4. DETERMINE RISK LEVEL
        # ============================================================

        if final_score >= self.high_risk_threshold:

            risk_level = RiskLevel.HIGH

        elif final_score >= self.medium_risk_threshold:

            risk_level = RiskLevel.MEDIUM

        else:

            risk_level = RiskLevel.LOW

        # ============================================================
        # 5. SAFETY RULE FOR EXITING WITHOUT A PERSON
        # ============================================================

        # Even if another scoring signal changes in the future,
        # an EXITING vehicle without a person association must not
        # automatically become a theft alert.

        if (
            vehicle_state == VehicleState.EXITING
            and person_id is None
        ):

            if risk_level == RiskLevel.HIGH:

                risk_level = RiskLevel.MEDIUM

            final_score = min(
                final_score,
                self.high_risk_threshold - 1,
            )

        # ============================================================
        # 6. NEW HIGH-RISK EVENT DETECTION
        # ============================================================

        alert_triggered = False

        if risk_level == RiskLevel.HIGH:

            # Only trigger an alert when the vehicle enters
            # HIGH-risk state.

            if not self._high_alert_active.get(
                vehicle_id,
                False,
            ):

                alert_triggered = True

                self._high_alert_active[
                    vehicle_id
                ] = True

        else:

            # Vehicle left HIGH-risk state.
            # Future HIGH-risk transition can create another alert.

            self._high_alert_active[
                vehicle_id
            ] = False

        # ============================================================
        # 7. EVIDENCE SNAPSHOT
        # ============================================================

        snapshot_file: Optional[str] = None

        if (
            alert_triggered
            and frame is not None
        ):

            last_shot = self._last_snapshot_time.get(
                vehicle_id,
                -100.0,
            )

            if (
                timestamp - last_shot
                >= self.snapshot_cooldown_sec
            ):

                snapshot_file = self._capture_snapshot(
                    frame=frame,
                    vehicle_id=vehicle_id,
                    person_id=person_id,
                    timestamp=timestamp,
                    risk_score=final_score,
                    vehicle_bbox=vehicle_bbox,
                )

                self._last_snapshot_time[
                    vehicle_id
                ] = timestamp

        # ============================================================
        # 8. CREATE RISK ASSESSMENT
        # ============================================================

        iso_time = datetime.now(
            timezone.utc
        ).isoformat()

        assessment = RiskAssessment(
            vehicle_id=vehicle_id,
            person_id=person_id,
            risk_score=final_score,
            risk_level=risk_level,
            reasons=reasons,
            timestamp=timestamp,
            iso_timestamp=iso_time,
            vehicle_state=vehicle_state,
            is_authorized=is_authorized,
            alert_triggered=alert_triggered,
            alert_label="POTENTIAL VEHICLE THEFT",
            snapshot_path=snapshot_file,
        )

        # ============================================================
        # 9. SAVE ONLY NEW HIGH-RISK ALERTS
        # ============================================================

        if alert_triggered:

            self.alerts.append(
                assessment
            )

            if self.database is not None:

                self.database.log_alert(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    vehicle_id=vehicle_id,
                    person_id=person_id,
                    risk_score=final_score,
                    risk_level=risk_level.value,
                    reasons=reasons,
                    snapshot_path=snapshot_file,
                )

        return assessment

    def _capture_snapshot(
        self,
        frame: np.ndarray,
        vehicle_id: int,
        person_id: Optional[int],
        timestamp: float,
        risk_score: int,
        vehicle_bbox: Optional[
            Tuple[int, int, int, int]
        ] = None,
    ) -> str:
        """Capture and save an annotated snapshot image as evidence."""

        snapshot_frame = frame.copy()

        # ============================================================
        # ALERT HEADER
        # ============================================================

        banner_h = 45

        overlay = snapshot_frame[
            :banner_h,
            :
        ].copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (
                snapshot_frame.shape[1],
                banner_h,
            ),
            (0, 0, 180),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.85,
            snapshot_frame[
                :banner_h,
                :
            ],
            0.15,
            0,
            snapshot_frame[
                :banner_h,
                :
            ],
        )

        title = (
            "[ALERT: POTENTIAL VEHICLE THEFT] "
            f"Vehicle #{vehicle_id} | "
            f"Risk: {risk_score} (HIGH)"
        )

        cv2.putText(
            snapshot_frame,
            title,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ============================================================
        # HIGHLIGHT VEHICLE
        # ============================================================

        if vehicle_bbox is not None:

            cv2.rectangle(
                snapshot_frame,
                (
                    vehicle_bbox[0],
                    vehicle_bbox[1],
                ),
                (
                    vehicle_bbox[2],
                    vehicle_bbox[3],
                ),
                (0, 0, 255),
                3,
            )

        # ============================================================
        # SAVE IMAGE
        # ============================================================

        filename = (
            f"alert_v{vehicle_id}_"
            f"p{person_id if person_id is not None else 'none'}_"
            f"{int(timestamp * 1000)}.jpg"
        )

        target_path = (
            self.snapshots_dir
            / filename
        )

        cv2.imwrite(
            str(target_path),
            snapshot_frame,
        )

        return str(target_path)

    def draw_risk_overlay(
        self,
        frame: np.ndarray,
        assessments: Dict[int, RiskAssessment],
        tracked_objects: List[TrackedObject],
    ) -> np.ndarray:
        """Annotate frame with risk scores and alert banners."""

        annotated = frame.copy()

        obj_map = {
            obj.track_id: obj
            for obj in tracked_objects
        }

        has_high_risk = False
        highest_score = 0
        top_alert_info = ""

        # ============================================================
        # DRAW RISK BADGES
        # ============================================================

        for v_id, assess in assessments.items():

            vehicle = obj_map.get(v_id)

            if not vehicle:
                continue

            # --------------------------------------------------------
            # Color by risk level
            # --------------------------------------------------------

            if assess.risk_level == RiskLevel.HIGH:

                tag_color = (0, 0, 255)

                has_high_risk = True

                if assess.risk_score > highest_score:

                    highest_score = assess.risk_score

                    top_alert_info = (
                        f"Vehicle #{v_id} "
                        f"(Score: {assess.risk_score})"
                    )

            elif assess.risk_level == RiskLevel.MEDIUM:

                tag_color = (0, 165, 255)

            else:

                tag_color = (0, 200, 0)

            # --------------------------------------------------------
            # Vehicle bounding box
            # --------------------------------------------------------

            x1, y1, x2, y2 = vehicle.bbox

            tag_text = (
                f"Risk: {assess.risk_score} "
                f"[{assess.risk_level.value}]"
            )

            (
                tw,
                th,
            ), _ = cv2.getTextSize(
                tag_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                1,
            )

            bx1 = x1

            by1 = y1 - th - 6

            if by1 < 30:

                by1 = y2 + 25

            # --------------------------------------------------------
            # Badge background
            # --------------------------------------------------------

            cv2.rectangle(
                annotated,
                (
                    bx1,
                    by1 - th - 2,
                ),
                (
                    bx1 + tw + 6,
                    by1 + 4,
                ),
                (20, 20, 20),
                -1,
            )

            cv2.rectangle(
                annotated,
                (
                    bx1,
                    by1 - th - 2,
                ),
                (
                    bx1 + tw + 6,
                    by1 + 4,
                ),
                tag_color,
                1,
            )

            cv2.putText(
                annotated,
                tag_text,
                (
                    bx1 + 3,
                    by1,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                tag_color,
                1,
                cv2.LINE_AA,
            )

        # ============================================================
        # HIGH-RISK BANNER
        # ============================================================

        if has_high_risk:

            banner_h = 35

            overlay = annotated[
                0:banner_h,
                :
            ].copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (
                    annotated.shape[1],
                    banner_h,
                ),
                (0, 0, 200),
                -1,
            )

            cv2.addWeighted(
                overlay,
                0.8,
                annotated[
                    0:banner_h,
                    :
                ],
                0.2,
                0,
                annotated[
                    0:banner_h,
                    :
                ],
            )

            alert_text = (
                "ALERT: POTENTIAL VEHICLE THEFT | "
                f"{top_alert_info}"
            )

            cv2.putText(
                annotated,
                alert_text,
                (15, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated

    def save_alerts_json(
        self,
        output_path: Path,
    ) -> None:
        """Save captured high-risk alert audit records to JSON."""

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = [
            alert.to_dict()
            for alert in self.alerts
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
        """Reset internal alert state."""

        self.alerts.clear()

        self._last_snapshot_time.clear()

        self._high_alert_active.clear()