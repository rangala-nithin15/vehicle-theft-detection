"""SQLite database module for persisting surveillance event history and theft risk alerts."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Union


class EventDatabase:
    """Manages SQLite persistence for vehicle state transitions and theft alerts."""

    def __init__(self, db_path: Union[str, Path] = "data/output/surveillance.db") -> None:
        """Initialize SQLite database manager.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get an active SQLite connection with row factory enabled."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema if tables do not already exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Table for vehicle temporal behavior transitions
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS behavior_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    frame_idx INTEGER NOT NULL,
                    vehicle_id INTEGER NOT NULL,
                    previous_state TEXT NOT NULL,
                    new_state TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Table for theft alerts
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    frame_idx INTEGER NOT NULL,
                    vehicle_id INTEGER NOT NULL,
                    person_id INTEGER,
                    risk_score REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    reasons TEXT NOT NULL,
                    snapshot_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def log_behavior_event(
        self,
        frame_idx: int,
        timestamp: Union[float, str],
        vehicle_id: int,
        previous_state: str,
        new_state: str,
        reason: str,
    ) -> int:
        """Log a vehicle behavior state transition event.
        
        Args:
            frame_idx: Video frame index.
            timestamp: Video timestamp in seconds or formatted string.
            vehicle_id: Track ID of the vehicle.
            previous_state: State prior to transition.
            new_state: State after transition.
            reason: Causal explanation for the transition.
            
        Returns:
            Inserted row ID.
        """
        ts_str = f"{timestamp:.2f}s" if isinstance(timestamp, (int, float)) else str(timestamp)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO behavior_events (timestamp, frame_idx, vehicle_id, previous_state, new_state, reason)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_str,
                    int(frame_idx),
                    int(vehicle_id),
                    str(previous_state),
                    str(new_state),
                    str(reason),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def log_alert(
        self,
        frame_idx: int,
        timestamp: Union[float, str],
        vehicle_id: int,
        person_id: Optional[int],
        risk_score: float,
        risk_level: str,
        reasons: Union[List[str], str],
        snapshot_path: Optional[str] = None,
    ) -> int:
        """Log a potential vehicle theft alert with contributing reasons and snapshot evidence.
        
        Args:
            frame_idx: Video frame index.
            timestamp: Video timestamp in seconds or string.
            vehicle_id: Track ID of the vehicle.
            person_id: Track ID of associated person, if any.
            risk_score: Calculated risk score (0-100).
            risk_level: 'LOW', 'MEDIUM', or 'HIGH'.
            reasons: Contributing risk signals.
            snapshot_path: File path to the captured evidence snapshot.
            
        Returns:
            Inserted row ID.
        """
        ts_str = f"{timestamp:.2f}s" if isinstance(timestamp, (int, float)) else str(timestamp)
        if isinstance(reasons, list):
            reasons_str = json.dumps(reasons)
        else:
            reasons_str = str(reasons)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO alerts (timestamp, frame_idx, vehicle_id, person_id, risk_score, risk_level, reasons, snapshot_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_str,
                    int(frame_idx),
                    int(vehicle_id),
                    int(person_id) if person_id is not None else None,
                    float(risk_score),
                    str(risk_level),
                    reasons_str,
                    str(snapshot_path or ""),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_recent_alerts(
        self,
        limit: int = 50,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent alerts, ordered latest first.
        
        Args:
            limit: Maximum records to return.
            min_score: Minimum risk score filter.
            
        Returns:
            List of alert dictionaries.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp, frame_idx, vehicle_id, person_id, risk_score, risk_level, reasons, snapshot_path, created_at
                FROM alerts
                WHERE risk_score >= ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (float(min_score), int(limit)),
            )
            rows = cursor.fetchall()
            results = []
            for row in rows:
                item = dict(row)
                try:
                    item["reasons"] = json.loads(item["reasons"])
                except Exception:
                    pass
                results.append(item)
            return results

    def get_recent_events(
        self,
        limit: int = 100,
        vehicle_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent behavior events, ordered latest first.
        
        Args:
            limit: Maximum records to return.
            vehicle_id: Optional filter for a specific vehicle track ID.
            
        Returns:
            List of event dictionaries.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if vehicle_id is not None:
                cursor.execute(
                    """
                    SELECT id, timestamp, frame_idx, vehicle_id, previous_state, new_state, reason, created_at
                    FROM behavior_events
                    WHERE vehicle_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (int(vehicle_id), int(limit)),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, timestamp, frame_idx, vehicle_id, previous_state, new_state, reason, created_at
                    FROM behavior_events
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_security_summary(self) -> Dict[str, int]:
        """Compute aggregate summary counts for the security status dashboard.
        
        Returns:
            Dictionary with count metrics.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM behavior_events")
            total_events = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM alerts")
            total_alerts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM alerts WHERE risk_score >= 70")
            high_risk_alerts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM alerts WHERE risk_score >= 40 AND risk_score < 70")
            medium_risk_alerts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT vehicle_id) FROM behavior_events")
            unique_vehicles_seen = cursor.fetchone()[0]

            return {
                "total_events": total_events,
                "total_alerts": total_alerts,
                "high_risk_alerts": high_risk_alerts,
                "medium_risk_alerts": medium_risk_alerts,
                "unique_vehicles_seen": unique_vehicles_seen,
            }

    def clear_history(self) -> None:
        """Clear all event and alert records from the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM behavior_events")
            cursor.execute("DELETE FROM alerts")
            conn.commit()
