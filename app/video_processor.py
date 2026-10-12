"""Video processor for handling detection, tracking, association, behavior analysis, and theft risk scoring."""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Union
import cv2

from app.association import AssociationEngine, PersonVehicleAssociation
from app.authorization import AuthorizationStore
from app.behavior import BehaviorEngine, VehicleState
from app.config import DetectionConfig, config as default_config
from app.database import EventDatabase
from app.detector import YOLODetector
from app.risk_engine import RiskAssessment, RiskEngine, RiskLevel
from app.tracker import ObjectTracker
from app.zone import ParkingZone

VEHICLE_CLASSES = {"car", "motorcycle", "bicycle", "truck", "bus"}


class VideoProcessor:
    """Manages reading, processing, visualizing, and writing video files across all phases."""

    def __init__(
        self,
        detector: Optional[YOLODetector] = None,
        tracker: Optional[ObjectTracker] = None,
        association_engine: Optional[AssociationEngine] = None,
        behavior_engine: Optional[BehaviorEngine] = None,
        auth_store: Optional[AuthorizationStore] = None,
        risk_engine: Optional[RiskEngine] = None,
        parking_zone: Optional[ParkingZone] = None,
        db: Optional[EventDatabase] = None,
        config: Optional[DetectionConfig] = None,
    ) -> None:
        """Initialize the video processor.
        
        Args:
            detector: Instance of YOLODetector.
            tracker: Instance of ObjectTracker.
            association_engine: Instance of AssociationEngine.
            behavior_engine: Instance of BehaviorEngine.
            auth_store: Instance of AuthorizationStore.
            risk_engine: Instance of RiskEngine.
            parking_zone: Instance of ParkingZone.
            db: Instance of EventDatabase for SQLite audit logging.
            config: Central DetectionConfig instance.
        """
        self.config = config or default_config
        self._detector = detector
        self._tracker = tracker
        self._association_engine = association_engine
        self._behavior_engine = behavior_engine
        self._auth_store = auth_store
        self._risk_engine = risk_engine
        self._parking_zone = parking_zone
        self._db = db

    @property
    def detector(self) -> YOLODetector:
        """Lazy-loaded detector instance."""
        if self._detector is None:
            self._detector = YOLODetector(config=self.config)
        return self._detector

    @property
    def tracker(self) -> ObjectTracker:
        """Lazy-loaded tracker instance."""
        if self._tracker is None:
            self._tracker = ObjectTracker(config=self.config)
        return self._tracker

    @property
    def parking_zone(self) -> ParkingZone:
        """Lazy-loaded parking zone instance."""
        if self._parking_zone is None:
            self._parking_zone = ParkingZone(self.config.parking_zone_points)
        return self._parking_zone

    @property
    def association_engine(self) -> AssociationEngine:
        """Lazy-loaded association engine instance."""
        if self._association_engine is None:
            self._association_engine = AssociationEngine(
                config=self.config,
                parking_zone=self.parking_zone,
            )
        return self._association_engine

    @property
    def behavior_engine(self) -> BehaviorEngine:
        """Lazy-loaded behavior engine instance."""
        if self._behavior_engine is None:
            self._behavior_engine = BehaviorEngine(
                config=self.config,
                parking_zone=self.parking_zone,
            )
        return self._behavior_engine

    @property
    def auth_store(self) -> AuthorizationStore:
        """Lazy-loaded authorization store instance."""
        if self._auth_store is None:
            if self.config.authorization_config_path.exists():
                self._auth_store = AuthorizationStore.from_yaml(self.config.authorization_config_path)
            else:
                self._auth_store = AuthorizationStore()
        return self._auth_store

    @property
    def db(self) -> EventDatabase:
        """Lazy-loaded SQLite event database instance."""
        if self._db is None:
            self._db = EventDatabase(self.config.db_path)
        return self._db

    @property
    def risk_engine(self) -> RiskEngine:
        """Lazy-loaded risk scoring engine instance."""
        if self._risk_engine is None:
            self._risk_engine = RiskEngine(
                config=self.config,
                auth_store=self.auth_store,
                snapshots_dir=self.config.snapshots_dir,
                database=self.db,
            )
        return self._risk_engine

    def process_video(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        show_display: bool = False,
    ) -> Dict[str, Any]:
        """Process a video file frame by frame through the object detector (Phase 1)."""
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input video file not found: {input_file.resolve()}")

        cap = cv2.VideoCapture(str(input_file))
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video stream from: {input_file.resolve()}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps is None or np_isnan(fps):
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        resolved_output_path = None
        if output_path is not None:
            resolved_output_path = Path(output_path)
            resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(resolved_output_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise IOError(f"Could not open VideoWriter for path: {resolved_output_path}")

        frames_processed = 0
        total_detections_count = 0
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frames_processed += 1

                # 1. Detect objects in current frame
                detections = self.detector.detect(frame)
                total_detections_count += len(detections)

                # 2. Draw bounding boxes and labels
                annotated_frame = self.detector.draw_detections(frame, detections)

                # 3. Informative HUD overlay
                hud_text = f"Frame: {frames_processed}/{total_frames if total_frames > 0 else '?'} | Detections: {len(detections)}"
                cv2.putText(
                    annotated_frame,
                    hud_text,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # 4. Save to output writer if requested
                if writer is not None:
                    writer.write(annotated_frame)

                # 5. Show frame if GUI display enabled
                if show_display:
                    cv2.imshow("Vehicle Detection - Phase 1", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show_display:
                cv2.destroyAllWindows()

        elapsed_time = max(time.time() - start_time, 1e-5)
        avg_fps = frames_processed / elapsed_time

        return {
            "input_path": str(input_file),
            "output_path": str(resolved_output_path) if resolved_output_path else None,
            "total_frames_processed": frames_processed,
            "total_detections": total_detections_count,
            "elapsed_seconds": round(elapsed_time, 2),
            "average_fps": round(avg_fps, 2),
        }

    def process_tracking(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        show_display: bool = False,
        draw_trail: bool = True,
    ) -> Dict[str, Any]:
        """Process a video file with multi-object tracking and persistent IDs (Phase 2)."""
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input video file not found: {input_file.resolve()}")

        cap = cv2.VideoCapture(str(input_file))
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video stream from: {input_file.resolve()}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps is None or np_isnan(fps):
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        resolved_output_path = None
        if output_path is not None:
            resolved_output_path = Path(output_path)
            resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(resolved_output_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise IOError(f"Could not open VideoWriter for path: {resolved_output_path}")

        self.tracker.reset()

        frames_processed = 0
        total_tracks_count = 0
        unique_persons: Set[int] = set()
        unique_vehicles: Set[int] = set()
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frames_processed += 1
                timestamp = frames_processed / fps

                # 1. Run multi-object tracking
                tracks = self.tracker.track(
                    frame=frame,
                    frame_idx=frames_processed,
                    timestamp=timestamp,
                )
                total_tracks_count += len(tracks)

                for trk in tracks:
                    if trk.class_name == "person":
                        unique_persons.add(trk.track_id)
                    elif trk.class_name in VEHICLE_CLASSES:
                        unique_vehicles.add(trk.track_id)

                # 2. Draw tracks with persistent IDs and trails
                annotated_frame = self.tracker.draw_tracks(
                    frame,
                    tracks,
                    draw_trail=draw_trail,
                )

                # 3. HUD overlay
                hud_text = (
                    f"Frame: {frames_processed}/{total_frames if total_frames > 0 else '?'} | "
                    f"Active: {len(tracks)} | "
                    f"Unique (Persons: {len(unique_persons)}, Vehicles: {len(unique_vehicles)})"
                )
                cv2.putText(
                    annotated_frame,
                    hud_text,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if writer is not None:
                    writer.write(annotated_frame)

                if show_display:
                    cv2.imshow("Multi-Object Tracking - Phase 2", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show_display:
                cv2.destroyAllWindows()

        elapsed_time = max(time.time() - start_time, 1e-5)
        avg_fps = frames_processed / elapsed_time
        all_unique_ids = self.tracker.seen_track_ids

        return {
            "input_path": str(input_file),
            "output_path": str(resolved_output_path) if resolved_output_path else None,
            "total_frames_processed": frames_processed,
            "total_tracks_count": total_tracks_count,
            "unique_track_ids_count": len(all_unique_ids),
            "unique_persons_count": len(unique_persons),
            "unique_vehicles_count": len(unique_vehicles),
            "elapsed_seconds": round(elapsed_time, 2),
            "average_fps": round(avg_fps, 2),
        }

    def process_association(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        show_display: bool = False,
        draw_trail: bool = True,
    ) -> Dict[str, Any]:
        """Process video with tracking, parking zones, and person-vehicle association (Phase 3)."""
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input video file not found: {input_file.resolve()}")

        cap = cv2.VideoCapture(str(input_file))
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video stream from: {input_file.resolve()}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps is None or np_isnan(fps):
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        resolved_output_path = None
        if output_path is not None:
            resolved_output_path = Path(output_path)
            resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(resolved_output_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise IOError(f"Could not open VideoWriter for path: {resolved_output_path}")

        self.tracker.reset()
        self.association_engine.reset()

        frames_processed = 0
        total_tracks_count = 0
        total_associations_count = 0
        persistent_associations_set: Set[Tuple[int, int]] = set()
        all_associations_set: Set[Tuple[int, int]] = set()
        unique_persons: Set[int] = set()
        unique_vehicles: Set[int] = set()
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frames_processed += 1
                timestamp = frames_processed / fps

                # 1. Run multi-object tracking
                tracks = self.tracker.track(
                    frame=frame,
                    frame_idx=frames_processed,
                    timestamp=timestamp,
                )
                total_tracks_count += len(tracks)

                for trk in tracks:
                    if trk.class_name == "person":
                        unique_persons.add(trk.track_id)
                    elif trk.class_name in VEHICLE_CLASSES:
                        unique_vehicles.add(trk.track_id)

                # 2. Run Person-Vehicle Association
                associations = self.association_engine.update(
                    tracked_objects=tracks,
                    frame_idx=frames_processed,
                    timestamp=timestamp,
                )
                total_associations_count += len(associations)

                for a in associations:
                    all_associations_set.add((a.person_id, a.vehicle_id))
                    if a.is_persistent:
                        persistent_associations_set.add((a.person_id, a.vehicle_id))

                # 3. Draw parking zone polygon
                annotated_frame = self.parking_zone.draw(frame)

                # 4. Draw tracks and motion trails
                annotated_frame = self.tracker.draw_tracks(
                    annotated_frame,
                    tracks,
                    draw_trail=draw_trail,
                )

                # 5. Draw person-vehicle associations and interaction badges
                annotated_frame = self.association_engine.draw_associations(
                    annotated_frame,
                    associations,
                    tracked_objects=tracks,
                )

                # 6. Informative HUD overlay
                persistent_now = sum(1 for a in associations if a.is_persistent)
                hud_line1 = (
                    f"Frame: {frames_processed}/{total_frames if total_frames > 0 else '?'} | "
                    f"Tracks: {len(tracks)} (P:{len(unique_persons)}, V:{len(unique_vehicles)})"
                )
                hud_line2 = (
                    f"Active Pairs: {len(associations)} | "
                    f"Persistent Interactions: {persistent_now} | "
                    f"Cumulative Persistent: {len(persistent_associations_set)}"
                )

                cv2.putText(
                    annotated_frame,
                    hud_line1,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated_frame,
                    hud_line2,
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 200, 255) if persistent_now == 0 else (0, 69, 255),
                    2,
                    cv2.LINE_AA,
                )

                if writer is not None:
                    writer.write(annotated_frame)

                if show_display:
                    cv2.imshow("Person-Vehicle Association - Phase 3", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show_display:
                cv2.destroyAllWindows()

        elapsed_time = max(time.time() - start_time, 1e-5)
        avg_fps = frames_processed / elapsed_time

        return {
            "input_path": str(input_file),
            "output_path": str(resolved_output_path) if resolved_output_path else None,
            "total_frames_processed": frames_processed,
            "total_tracks_count": total_tracks_count,
            "unique_persons_count": len(unique_persons),
            "unique_vehicles_count": len(unique_vehicles),
            "unique_associations_count": len(all_associations_set),
            "persistent_associations_count": len(persistent_associations_set),
            "elapsed_seconds": round(elapsed_time, 2),
            "average_fps": round(avg_fps, 2),
        }

    def process_behavior(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        events_output_path: Optional[Union[str, Path]] = None,
        show_display: bool = False,
        draw_trail: bool = True,
    ) -> Dict[str, Any]:
        """Process video with tracking, association, and temporal vehicle behavior state machine (Phase 4)."""
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input video file not found: {input_file.resolve()}")

        cap = cv2.VideoCapture(str(input_file))
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video stream from: {input_file.resolve()}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps is None or np_isnan(fps):
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        resolved_output_path = None
        if output_path is not None:
            resolved_output_path = Path(output_path)
            resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(resolved_output_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise IOError(f"Could not open VideoWriter for path: {resolved_output_path}")

        self.tracker.reset()
        self.association_engine.reset()
        self.behavior_engine.reset()

        frames_processed = 0
        total_tracks_count = 0
        unique_persons: Set[int] = set()
        unique_vehicles: Set[int] = set()
        latest_vehicle_states: Dict[int, VehicleState] = {}
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frames_processed += 1
                timestamp = frames_processed / fps

                # 1. Multi-object tracking
                tracks = self.tracker.track(
                    frame=frame,
                    frame_idx=frames_processed,
                    timestamp=timestamp,
                )
                total_tracks_count += len(tracks)

                for trk in tracks:
                    if trk.class_name == "person":
                        unique_persons.add(trk.track_id)
                    elif trk.class_name in VEHICLE_CLASSES:
                        unique_vehicles.add(trk.track_id)

                # 2. Person-Vehicle Association
                associations = self.association_engine.update(
                    tracked_objects=tracks,
                    frame_idx=frames_processed,
                    timestamp=timestamp,
                )

                # 3. Temporal Behavior State Analysis
                latest_vehicle_states = self.behavior_engine.update(
                    tracked_objects=tracks,
                    associations=associations,
                    frame_idx=frames_processed,
                    timestamp=timestamp,
                )

                # 4. Render Parking Zone
                annotated_frame = self.parking_zone.draw(frame)

                # 5. Render Tracks and Motion Trails
                annotated_frame = self.tracker.draw_tracks(
                    annotated_frame,
                    tracks,
                    draw_trail=draw_trail,
                )

                # 6. Render Associations
                annotated_frame = self.association_engine.draw_associations(
                    annotated_frame,
                    associations,
                    tracked_objects=tracks,
                )

                # 7. Render Vehicle Behavior State Badges
                annotated_frame = self.behavior_engine.draw_vehicle_states(
                    annotated_frame,
                    latest_vehicle_states,
                    tracked_objects=tracks,
                )

                # 8. HUD Overlay
                states_summary = ", ".join(
                    f"V#{vid}:{state.value}" for vid, state in latest_vehicle_states.items()
                ) or "None"
                hud_line1 = (
                    f"Frame: {frames_processed}/{total_frames if total_frames > 0 else '?'} | "
                    f"Tracks: {len(tracks)} (P:{len(unique_persons)}, V:{len(unique_vehicles)})"
                )
                hud_line2 = (
                    f"Events: {len(self.behavior_engine.events)} | "
                    f"Vehicle States: {states_summary[:50]}"
                )

                cv2.putText(
                    annotated_frame,
                    hud_line1,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated_frame,
                    hud_line2,
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )

                if writer is not None:
                    writer.write(annotated_frame)

                if show_display:
                    cv2.imshow("Vehicle Behavior Analysis - Phase 4", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show_display:
                cv2.destroyAllWindows()

        if events_output_path is not None:
            self.behavior_engine.save_events_json(Path(events_output_path))

        elapsed_time = max(time.time() - start_time, 1e-5)
        avg_fps = frames_processed / elapsed_time

        return {
            "input_path": str(input_file),
            "output_path": str(resolved_output_path) if resolved_output_path else None,
            "events_path": str(events_output_path) if events_output_path else None,
            "total_frames_processed": frames_processed,
            "total_tracks_count": total_tracks_count,
            "unique_persons_count": len(unique_persons),
            "unique_vehicles_count": len(unique_vehicles),
            "total_transition_events": len(self.behavior_engine.events),
            "final_vehicle_states": {
                vid: state.value for vid, state in latest_vehicle_states.items()
            },
            "elapsed_seconds": round(elapsed_time, 2),
            "average_fps": round(avg_fps, 2),
        }

    def process_risk_assessment(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        alerts_output_path: Optional[Union[str, Path]] = None,
        events_output_path: Optional[Union[str, Path]] = None,
        show_display: bool = False,
        draw_trail: bool = True,
    ) -> Dict[str, Any]:
        """Process video with authorization checks, behavior analysis, and theft risk scoring (Phase 5).
        
        Args:
            input_path: Path to the input video file.
            output_path: Optional path to save the annotated output video.
            alerts_output_path: Optional path to save structured high-risk alerts JSON.
            events_output_path: Optional path to save state transition events JSON.
            show_display: Whether to display frames in an OpenCV window during processing.
            draw_trail: Whether to draw motion trajectory trails.
            
        Returns:
            Dictionary containing comprehensive pipeline evaluation statistics.
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input video file not found: {input_file.resolve()}")

        cap = cv2.VideoCapture(str(input_file))
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video stream from: {input_file.resolve()}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps is None or np_isnan(fps):
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        resolved_output_path = None
        if output_path is not None:
            resolved_output_path = Path(output_path)
            resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(resolved_output_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise IOError(f"Could not open VideoWriter for path: {resolved_output_path}")

        # Reset states for fresh video run
        self.tracker.reset()
        self.association_engine.reset()
        self.behavior_engine.reset()
        self.risk_engine.reset()

        frames_processed = 0
        total_tracks_count = 0
        logged_event_count = 0
        unique_persons: Set[int] = set()
        unique_vehicles: Set[int] = set()
        highest_risk_score = 0
        latest_assessments: Dict[int, RiskAssessment] = {}
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frames_processed += 1
                timestamp = frames_processed / fps

                # 1. Multi-object tracking
                tracks = self.tracker.track(
                    frame=frame,
                    frame_idx=frames_processed,
                    timestamp=timestamp,
                )
                total_tracks_count += len(tracks)

                for trk in tracks:
                    if trk.class_name == "person":
                        unique_persons.add(trk.track_id)
                    elif trk.class_name in VEHICLE_CLASSES:
                        unique_vehicles.add(trk.track_id)

                # 2. Person-Vehicle Association
                associations = self.association_engine.update(
                    tracked_objects=tracks,
                    frame_idx=frames_processed,
                    timestamp=timestamp,
                )

                # 3. Temporal Behavior State Analysis
                vehicle_states = self.behavior_engine.update(
                    tracked_objects=tracks,
                    associations=associations,
                    frame_idx=frames_processed,
                    timestamp=timestamp,
                )

                # Log any new transition events to SQLite
                while logged_event_count < len(self.behavior_engine.events):
                    ev = self.behavior_engine.events[logged_event_count]
                    self.db.log_behavior_event(
                        frame_idx=frames_processed,
                        timestamp=ev.timestamp,
                        vehicle_id=ev.vehicle_id,
                        previous_state=ev.previous_state.value,
                        new_state=ev.new_state.value,
                        reason=ev.reason,
                    )
                    logged_event_count += 1

                # 4. Multi-Signal Theft Risk Evaluation
                v_map = {t.track_id: t for t in tracks if t.class_name in VEHICLE_CLASSES}
                current_assessments: Dict[int, RiskAssessment] = {}

                for vid, state in vehicle_states.items():
                    v_obj = v_map.get(vid)
                    p_id = (
                        self.behavior_engine.vehicles[vid].associated_person_id
                        if vid in self.behavior_engine.vehicles
                        else None
                    )
                    v_bbox = v_obj.bbox if v_obj else None

                    assessment = self.risk_engine.evaluate(
                        vehicle_id=vid,
                        vehicle_state=state,
                        person_id=p_id,
                        timestamp=timestamp,
                        frame=frame,
                        vehicle_bbox=v_bbox,
                        frame_idx=frames_processed,
                    )
                    current_assessments[vid] = assessment
                    if assessment.risk_score > highest_risk_score:
                        highest_risk_score = assessment.risk_score

                latest_assessments = current_assessments

                # 5. Render Parking Zone
                annotated_frame = self.parking_zone.draw(frame)

                # 6. Render Tracks and Motion Trails
                annotated_frame = self.tracker.draw_tracks(
                    annotated_frame,
                    tracks,
                    draw_trail=draw_trail,
                )

                # 7. Render Associations
                annotated_frame = self.association_engine.draw_associations(
                    annotated_frame,
                    associations,
                    tracked_objects=tracks,
                )

                # 8. Render Vehicle Behavior State Badges
                annotated_frame = self.behavior_engine.draw_vehicle_states(
                    annotated_frame,
                    vehicle_states,
                    tracked_objects=tracks,
                )

                # 9. Render Risk Scores and Potential Theft Alerts
                annotated_frame = self.risk_engine.draw_risk_overlay(
                    annotated_frame,
                    current_assessments,
                    tracked_objects=tracks,
                )

                # 10. HUD Overlay
                alerts_count = len(self.risk_engine.alerts)
                hud_line1 = (
                    f"Frame: {frames_processed}/{total_frames if total_frames > 0 else '?'} | "
                    f"Tracks: {len(tracks)} (P:{len(unique_persons)}, V:{len(unique_vehicles)})"
                )
                hud_line2 = (
                    f"High Risk Alerts: {alerts_count} | "
                    f"Peak Risk: {highest_risk_score} | "
                    f"Events: {len(self.behavior_engine.events)}"
                )

                cv2.putText(
                    annotated_frame,
                    hud_line1,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated_frame,
                    hud_line2,
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 0, 255) if alerts_count > 0 else (0, 200, 0),
                    2,
                    cv2.LINE_AA,
                )

                if writer is not None:
                    writer.write(annotated_frame)

                if show_display:
                    cv2.imshow("Theft Risk Analysis - Phase 5", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show_display:
                cv2.destroyAllWindows()

        # Save alert records and events JSON
        if alerts_output_path is not None:
            self.risk_engine.save_alerts_json(Path(alerts_output_path))
        if events_output_path is not None:
            self.behavior_engine.save_events_json(Path(events_output_path))

        elapsed_time = max(time.time() - start_time, 1e-5)
        avg_fps = frames_processed / elapsed_time

        return {
            "input_path": str(input_file),
            "output_path": str(resolved_output_path) if resolved_output_path else None,
            "alerts_path": str(alerts_output_path) if alerts_output_path else None,
            "events_path": str(events_output_path) if events_output_path else None,
            "total_frames_processed": frames_processed,
            "total_tracks_count": total_tracks_count,
            "unique_persons_count": len(unique_persons),
            "unique_vehicles_count": len(unique_vehicles),
            "total_transition_events": len(self.behavior_engine.events),
            "total_high_risk_alerts": len(self.risk_engine.alerts),
            "highest_risk_score": highest_risk_score,
            "final_assessments": {
                vid: a.to_dict() for vid, a in latest_assessments.items()
            },
            "elapsed_seconds": round(elapsed_time, 2),
            "average_fps": round(avg_fps, 2),
        }

    def stream_risk_assessment(
        self,
        input_path: Union[str, Path],
        max_frames: Optional[int] = None,
        frame_stride: int = 1,
        draw_trail: bool = True,
        save_to_db: bool = True,
    ):
        """Generator that yields per-frame surveillance results for live dashboard streaming.
        
        Args:
            input_path: Path to input video file.
            max_frames: Optional max number of frames to process.
            frame_stride: Process every Nth frame (1 = every frame).
            draw_trail: Whether to draw motion trajectory trails.
            save_to_db: Whether to write events and alerts into SQLite.
            
        Yields:
            Dictionary containing:
                - frame_bgr: raw annotated frame (OpenCV BGR)
                - frame_rgb: RGB frame ready for Streamlit display
                - frame_idx: current frame index
                - total_frames: total frames in video
                - timestamp: current video time in seconds
                - tracks: list of active TrackedObject
                - associations: list of active PersonVehicleAssociation
                - vehicle_states: dict of vehicle_id -> VehicleState
                - assessments: dict of vehicle_id -> RiskAssessment
                - new_alerts: list of any alerts triggered this frame
                - fps: rolling processing FPS
                - unique_persons: set of seen person IDs
                - unique_vehicles: set of seen vehicle IDs
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input video file not found: {input_file.resolve()}")

        cap = cv2.VideoCapture(str(input_file))
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video stream from: {input_file.resolve()}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps is None or np_isnan(fps):
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Reset components
        self.tracker.reset()
        self.association_engine.reset()
        self.behavior_engine.reset()
        self.risk_engine.reset()

        frames_read = 0
        frames_processed = 0
        logged_event_count = 0
        unique_persons: Set[int] = set()
        unique_vehicles: Set[int] = set()
        start_time = time.time()
        prev_alert_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frames_read += 1
                if frame_stride > 1 and (frames_read % frame_stride != 0):
                    continue

                frames_processed += 1
                if max_frames and frames_processed > max_frames:
                    break

                timestamp = frames_read / fps

                # 1. Multi-object tracking
                tracks = self.tracker.track(
                    frame=frame,
                    frame_idx=frames_read,
                    timestamp=timestamp,
                )

                for trk in tracks:
                    if trk.class_name == "person":
                        unique_persons.add(trk.track_id)
                    elif trk.class_name in VEHICLE_CLASSES:
                        unique_vehicles.add(trk.track_id)

                # 2. Person-Vehicle Association
                associations = self.association_engine.update(
                    tracked_objects=tracks,
                    frame_idx=frames_read,
                    timestamp=timestamp,
                )

                # 3. Temporal Behavior State Analysis
                vehicle_states = self.behavior_engine.update(
                    tracked_objects=tracks,
                    associations=associations,
                    frame_idx=frames_read,
                    timestamp=timestamp,
                )

                # SQLite event logging
                if save_to_db:
                    while logged_event_count < len(self.behavior_engine.events):
                        ev = self.behavior_engine.events[logged_event_count]
                        self.db.log_behavior_event(
                            frame_idx=frames_read,
                            timestamp=ev.timestamp,
                            vehicle_id=ev.vehicle_id,
                            previous_state=ev.previous_state.value,
                            new_state=ev.new_state.value,
                            reason=ev.reason,
                        )
                        logged_event_count += 1

                # 4. Multi-Signal Theft Risk Evaluation
                v_map = {t.track_id: t for t in tracks if t.class_name in VEHICLE_CLASSES}
                current_assessments: Dict[int, RiskAssessment] = {}

                for vid, state in vehicle_states.items():
                    v_obj = v_map.get(vid)
                    p_id = (
                        self.behavior_engine.vehicles[vid].associated_person_id
                        if vid in self.behavior_engine.vehicles
                        else None
                    )
                    v_bbox = v_obj.bbox if v_obj else None

                    assessment = self.risk_engine.evaluate(
                        vehicle_id=vid,
                        vehicle_state=state,
                        person_id=p_id,
                        timestamp=timestamp,
                        frame=frame,
                        vehicle_bbox=v_bbox,
                        frame_idx=frames_read,
                    )
                    current_assessments[vid] = assessment

                # Identify newly triggered alerts
                current_alert_count = len(self.risk_engine.alerts)
                new_alerts = (
                    self.risk_engine.alerts[prev_alert_count:current_alert_count]
                    if current_alert_count > prev_alert_count
                    else []
                )
                prev_alert_count = current_alert_count

                # Render overlays
                annotated_frame = self.parking_zone.draw(frame)
                annotated_frame = self.tracker.draw_tracks(
                    annotated_frame,
                    tracks,
                    draw_trail=draw_trail,
                )
                annotated_frame = self.association_engine.draw_associations(
                    annotated_frame,
                    associations,
                    tracked_objects=tracks,
                )
                annotated_frame = self.behavior_engine.draw_vehicle_states(
                    annotated_frame,
                    vehicle_states,
                    tracked_objects=tracks,
                )
                annotated_frame = self.risk_engine.draw_risk_overlay(
                    annotated_frame,
                    current_assessments,
                    tracked_objects=tracks,
                )

                # HUD Overlay
                alerts_count = len(self.risk_engine.alerts)
                hud_line1 = f"Frame: {frames_read}/{total_frames if total_frames > 0 else '?'} | Tracks: {len(tracks)} (P:{len(unique_persons)}, V:{len(unique_vehicles)})"
                hud_line2 = f"Alerts: {alerts_count} | Events: {len(self.behavior_engine.events)}"

                cv2.putText(
                    annotated_frame,
                    hud_line1,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated_frame,
                    hud_line2,
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 0, 255) if alerts_count > 0 else (0, 200, 0),
                    2,
                    cv2.LINE_AA,
                )

                elapsed = max(time.time() - start_time, 1e-5)
                current_fps = frames_processed / elapsed
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                yield {
                    "frame_bgr": annotated_frame,
                    "frame_rgb": frame_rgb,
                    "frame_idx": frames_read,
                    "total_frames": total_frames,
                    "timestamp": timestamp,
                    "tracks": tracks,
                    "associations": associations,
                    "vehicle_states": vehicle_states,
                    "assessments": current_assessments,
                    "new_alerts": new_alerts,
                    "fps": round(current_fps, 1),
                    "unique_persons": set(unique_persons),
                    "unique_vehicles": set(unique_vehicles),
                }

        finally:
            cap.release()


def np_isnan(val: float) -> bool:
    """Helper to check if a float is NaN without requiring extra imports."""
    return val != val

