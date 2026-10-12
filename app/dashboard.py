"""Streamlit Surveillance Dashboard for Vehicle Theft Detection & Security Monitoring.

Launch command:
    streamlit run app/dashboard.py
"""

from datetime import datetime
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import streamlit as st

from app.behavior import VehicleState
from app.config import DetectionConfig, config as default_config
from app.database import EventDatabase
from app.detector import YOLODetector
from app.risk_engine import RiskLevel
from app.video_processor import VideoProcessor

# Configure page layout and visual appearance
st.set_page_config(
    page_title="Vehicle Theft Detection Surveillance System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for crisp surveillance UI styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 8px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        color: #f8fafc;
        font-weight: 700;
    }
    .alert-card-high {
        background-color: #2a1215;
        border: 1px solid #ef4444;
        border-left: 6px solid #ef4444;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .alert-card-med {
        background-color: #2b1f11;
        border: 1px solid #f59e0b;
        border-left: 6px solid #f59e0b;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-high {
        background-color: #dc2626;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-medium {
        background-color: #d97706;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-low {
        background-color: #16a34a;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_database() -> EventDatabase:
    """Singleton connection to the local SQLite database."""
    return EventDatabase(default_config.db_path)


@st.cache_resource
def get_processor() -> VideoProcessor:
    """Singleton instance of the computer vision pipeline processor."""
    return VideoProcessor(config=default_config)


def main() -> None:
    """Main dashboard render entrypoint."""
    db = get_database()
    processor = get_processor()

    # Header & Research Positioning
    st.title("🛡️ Contextual Vehicle Theft Detection Surveillance System")
    st.markdown(
        "> **Research Concept**: *Existing video anomaly detection approaches primarily identify "
        "anomalous activities in surveillance scenes. This project extends the concept toward "
        "vehicle-specific contextual reasoning by associating people with individual parked vehicles "
        "and analyzing authorization, temporal interaction, vehicle movement, and exit behavior to "
        "estimate vehicle-theft risk.*"
    )

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Surveillance Control Panel")

        # Video source selection
        st.subheader("1. Video Source")
        video_source_type = st.radio(
            "Select Input Source",
            ["Sample Video", "Upload Video", "Custom File Path"],
            index=0,
        )

        selected_video_path: Optional[Path] = None

        if video_source_type == "Sample Video":
            input_dir = default_config.input_dir
            sample_videos = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.avi"))
            if sample_videos:
                chosen_sample = st.selectbox(
                    "Choose sample video",
                    options=[str(p) for p in sample_videos],
                    index=0,
                )
                selected_video_path = Path(chosen_sample) if chosen_sample else None
            else:
                st.warning("No sample videos found in data/input/. Please upload one.")

        elif video_source_type == "Upload Video":
            uploaded_file = st.file_uploader(
                "Upload surveillance footage",
                type=["mp4", "avi", "mov", "mkv"],
            )
            if uploaded_file is not None:
                # Save uploaded file to temporary directory
                temp_dir = Path("data/input")
                temp_dir.mkdir(parents=True, exist_ok=True)
                save_path = temp_dir / f"uploaded_{uploaded_file.name}"
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.read())
                selected_video_path = save_path
                st.success(f"Uploaded: {uploaded_file.name}")

        else:
            custom_input = st.text_input(
                "Video File Path",
                value="data/input/test.mp4",
            )
            if custom_input and Path(custom_input).exists():
                selected_video_path = Path(custom_input)
            elif custom_input:
                st.error(f"Path not found: {custom_input}")

        # Analysis Settings
        st.subheader("2. Processing Parameters")
        max_frames_choice = st.slider(
            "Frame Limit (0 = Full Video)",
            min_value=0,
            max_value=1000,
            value=0,
            step=50,
            help="Limit processed frames for fast previews",
        )
        frame_stride = st.slider(
            "Frame Stride",
            min_value=1,
            max_value=5,
            value=1,
            help="1 = process every frame, 2 = every 2nd frame (faster)",
        )
        conf_thresh = st.slider(
            "YOLO Confidence Threshold",
            min_value=0.15,
            max_value=0.90,
            value=default_config.confidence_threshold,
            step=0.05,
        )
        save_video_output = st.checkbox("Save Annotated Output Video", value=True)

        st.subheader("3. Database Actions")
        if st.button("🗑️ Clear History & Alerts", use_container_width=True):
            db.clear_history()
            st.success("Database history cleared!")

    # Top Row: Security Status Overview Metrics
    db_summary = db.get_security_summary()
    st.subheader("📊 Live Security Status Overview")

    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5, col_kpi6, col_kpi7 = st.columns(7)

    # Session stats containers
    total_persons_placeholder = col_kpi1.empty()
    total_vehicles_placeholder = col_kpi2.empty()
    parked_placeholder = col_kpi3.empty()
    interacting_placeholder = col_kpi4.empty()
    moving_placeholder = col_kpi5.empty()
    suspicious_placeholder = col_kpi6.empty()
    high_risk_placeholder = col_kpi7.empty()

    # Initial default metric values
    total_persons_placeholder.metric("Total Persons", "0")
    total_vehicles_placeholder.metric("Total Vehicles", db_summary.get("unique_vehicles_seen", 0))
    parked_placeholder.metric("Parked Vehicles", "0")
    interacting_placeholder.metric("Interacting", "0")
    moving_placeholder.metric("Moving Vehicles", "0")
    suspicious_placeholder.metric("Suspicious", db_summary.get("medium_risk_alerts", 0))
    high_risk_placeholder.metric("🚨 High Risk", db_summary.get("high_risk_alerts", 0))

    # Main Screen: 2 Columns (Left: Detection View, Right: Relationships & Telemetry)
    st.markdown("---")
    col_video, col_relationships = st.columns([3, 2])

    with col_video:
        st.subheader("📹 Real-Time Detection & Tracking Stream")
        video_placeholder = st.empty()
        status_bar = st.empty()
        progress_bar = st.empty()

    with col_relationships:
        st.subheader("🔗 Active Person–Vehicle Relationships")
        relationships_placeholder = st.empty()

        st.subheader("🚗 Monitored Vehicles State Monitor")
        vehicle_states_placeholder = st.empty()

    # Run Analysis Trigger
    start_analysis = st.sidebar.button("▶️ Start Surveillance Analysis", type="primary", use_container_width=True)

    if start_analysis:
        if not selected_video_path or not selected_video_path.exists():
            st.error("Please provide a valid input video file before starting.")
            return

        # Prepare output path
        output_video_path = None
        if save_video_output:
            default_config.output_dir.mkdir(parents=True, exist_ok=True)
            output_video_path = default_config.output_dir / "dashboard_annotated.mp4"

        # Apply user config overrides
        processor.config.confidence_threshold = conf_thresh
        writer = None

        if output_video_path:
            cap_probe = cv2.VideoCapture(str(selected_video_path))
            probe_w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
            probe_h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
            probe_fps = cap_probe.get(cv2.CAP_PROP_FPS) or 25.0
            cap_probe.release()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_video_path), fourcc, probe_fps, (probe_w, probe_h))

        # Stream pipeline execution
        max_frames_limit = max_frames_choice if max_frames_choice > 0 else None
        stream = processor.stream_risk_assessment(
            input_path=selected_video_path,
            max_frames=max_frames_limit,
            frame_stride=frame_stride,
            draw_trail=True,
            save_to_db=True,
        )

        all_unique_persons = set()
        all_unique_vehicles = set()
        last_associations_rendered = []
        suspicious_count = 0
        high_risk_count = 0

        for frame_data in stream:
            # 1. Update Video View
            frame_rgb = frame_data["frame_rgb"]
            video_placeholder.image(
                frame_rgb,
                channels="RGB",
                use_container_width=True,
            )

            # Write to output file if enabled
            if writer is not None:
                writer.write(frame_data["frame_bgr"])

            # 2. Update Progress & Telemetry
            cur_idx = frame_data["frame_idx"]
            tot_frames = frame_data["total_frames"]
            cur_fps = frame_data["fps"]
            pct = min(cur_idx / tot_frames, 1.0) if tot_frames > 0 else 0.0
            progress_bar.progress(pct)
            status_bar.caption(
                f"**Processing Frame:** {cur_idx}/{tot_frames} | **Timestamp:** {frame_data['timestamp']:.2f}s | "
                f"**Speed:** {cur_fps} FPS | **Active Tracks:** {len(frame_data['tracks'])}"
            )

            # 3. Update Status Metrics
            v_states = frame_data["vehicle_states"]
            parked_count = sum(1 for s in v_states.values() if s == VehicleState.PARKED)
            interacting_count = sum(1 for s in v_states.values() if s in {VehicleState.INTERACTION, VehicleState.ACCESS_ATTEMPT})
            moving_count = sum(1 for s in v_states.values() if s in {VehicleState.VEHICLE_MOVING, VehicleState.EXITING})

            all_unique_persons.update(frame_data["unique_persons"])
            all_unique_vehicles.update(frame_data["unique_vehicles"])

            # Check assessments
            for assess in frame_data["assessments"].values():
                if assess.risk_level == RiskLevel.HIGH:
                    high_risk_count += 1
                elif assess.risk_level == RiskLevel.MEDIUM:
                    suspicious_count += 1

            total_persons_placeholder.metric("Total Persons", len(all_unique_persons))
            total_vehicles_placeholder.metric("Total Vehicles", len(all_unique_vehicles))
            parked_placeholder.metric("Parked Vehicles", parked_count)
            interacting_placeholder.metric("Interacting", interacting_count)
            moving_placeholder.metric("Moving Vehicles", moving_count)
            suspicious_placeholder.metric("Suspicious", suspicious_count)
            high_risk_placeholder.metric("🚨 High Risk", len(processor.risk_engine.alerts))

            # 4. Render Active Relationships Panel
            associations = frame_data["associations"]
            if associations:
                assoc_html = ""
                for assoc in associations:
                    assoc_html += (
                        f"<div class='metric-card'>"
                        f"<strong>Person #{assoc.person_id}</strong> &nbsp;➔&nbsp; "
                        f"<strong>Vehicle #{assoc.vehicle_id}</strong><br>"
                        f"<small style='color: #cbd5e1;'>Distance: {assoc.distance_px:.1f}px | "
                        f"Persistence: {assoc.persistence_frames} frames ({assoc.persistence_seconds:.1f}s) | "
                        f"Status: {assoc.status.value}</small>"
                        f"</div>"
                    )
                relationships_placeholder.markdown(assoc_html, unsafe_allow_html=True)
            else:
                relationships_placeholder.info("No active person-vehicle associations detected.")

            # 5. Render Monitored Vehicles State Monitor
            if v_states:
                state_html = "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 8px;'>"
                for vid, st_val in v_states.items():
                    color = "#22c55e" if st_val == VehicleState.PARKED else ("#ef4444" if st_val in {VehicleState.ACCESS_ATTEMPT, VehicleState.EXITING} else "#eab308")
                    state_html += (
                        f"<div style='background-color:#1e293b; padding:8px; border-radius:6px; border-left:4px solid {color};'>"
                        f"<strong>Vehicle #{vid}</strong><br>"
                        f"<small style='color:{color}; font-weight:600;'>{st_val.value}</small>"
                        f"</div>"
                    )
                state_html += "</div>"
                vehicle_states_placeholder.markdown(state_html, unsafe_allow_html=True)
            else:
                vehicle_states_placeholder.info("No active vehicles being tracked in frame.")

        if writer is not None:
            writer.release()

        status_bar.success("Surveillance analysis complete!")
        progress_bar.progress(1.0)

    # Alert Panel & Evidence Snapshots
    st.markdown("---")
    st.subheader("🚨 Security Alerts & Evidence Snapshots")

    alerts = db.get_recent_alerts(limit=20, min_score=0.0)
    if alerts:
        for alert in alerts:
            is_high = alert["risk_score"] >= 70
            badge_class = "badge-high" if is_high else "badge-medium"
            card_class = "alert-card-high" if is_high else "alert-card-med"
            prefix = "🚨 POTENTIAL VEHICLE THEFT" if is_high else "⚠️ SUSPICIOUS ACTIVITY"

            col_alert_info, col_alert_img = st.columns([3, 1])

            with col_alert_info:
                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <h4>{prefix} &nbsp; <span class="{badge_class}">Score: {int(alert['risk_score'])}/100 [{alert['risk_level']}]</span></h4>
                        <p><strong>Timestamp:</strong> {alert['timestamp']} &nbsp;|&nbsp; <strong>Frame:</strong> {alert['frame_idx']}</p>
                        <p><strong>Vehicle ID:</strong> Vehicle #{alert['vehicle_id']} &nbsp;|&nbsp; <strong>Associated Person:</strong> Person #{alert['person_id'] or 'None'}</p>
                        <p><strong>Contributing Risk Reasons:</strong></p>
                        <ul>
                            {''.join(f'<li>{r}</li>' for r in (alert['reasons'] if isinstance(alert['reasons'], list) else [alert['reasons']]))}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_alert_img:
                snap_path = alert.get("snapshot_path")
                if snap_path and Path(snap_path).exists():
                    st.image(snap_path, caption=f"Evidence Snapshot #{alert['id']}", use_container_width=True)
                else:
                    st.caption("No snapshot image available.")
    else:
        st.info("No high-risk security alerts recorded in the database.")

    # Event History & Database Audit Log
    st.markdown("---")
    st.subheader("📜 Event History & SQLite Audit Trail")

    tab_events, tab_alerts = st.tabs(["Vehicle State Transitions", "Raw Alert Logs"])

    with tab_events:
        filter_col1, filter_col2 = st.columns([2, 2])
        v_filter = filter_col1.number_input("Filter by Vehicle ID (0 = All)", min_value=0, value=0, step=1)
        limit_events = filter_col2.slider("Max Events", min_value=10, max_value=200, value=50, step=10)

        filter_vid = int(v_filter) if v_filter > 0 else None
        event_rows = db.get_recent_events(limit=limit_events, vehicle_id=filter_vid)

        if event_rows:
            st.dataframe(
                event_rows,
                column_config={
                    "id": "Log ID",
                    "timestamp": "Timestamp",
                    "frame_idx": "Frame",
                    "vehicle_id": "Vehicle ID",
                    "previous_state": "From State",
                    "new_state": "To State",
                    "reason": "Transition Reason",
                    "created_at": "Logged At",
                },
                use_container_width=True,
            )
        else:
            st.caption("No vehicle behavior transition events found in database.")

    with tab_alerts:
        raw_alerts = db.get_recent_alerts(limit=50, min_score=0.0)
        if raw_alerts:
            # Flatten reasons for display in table
            table_data = []
            for a in raw_alerts:
                item = dict(a)
                if isinstance(item.get("reasons"), list):
                    item["reasons"] = "; ".join(item["reasons"])
                table_data.append(item)
            st.dataframe(
                table_data,
                column_config={
                    "id": "Alert ID",
                    "timestamp": "Time",
                    "frame_idx": "Frame",
                    "vehicle_id": "Vehicle ID",
                    "person_id": "Person ID",
                    "risk_score": "Risk Score",
                    "risk_level": "Level",
                    "reasons": "Risk Reasons",
                    "snapshot_path": "Snapshot File",
                },
                use_container_width=True,
            )
        else:
            st.caption("No alert audit logs found.")


if __name__ == "__main__":
    main()
