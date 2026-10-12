"""CLI script to run authorization-aware vehicle theft risk analysis (Phase 5)."""

import argparse
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.association import AssociationEngine
from app.authorization import AuthorizationStore
from app.behavior import BehaviorEngine
from app.config import DetectionConfig
from app.database import EventDatabase
from app.risk_engine import RiskEngine
from app.tracker import ObjectTracker
from app.video_processor import VideoProcessor
from app.zone import ParkingZone


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Authorization and Vehicle Theft Risk Analysis (Phase 5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to input video file (e.g. data/input/test.mp4)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save annotated video (e.g. data/output/risk_analyzed.mp4)",
    )
    parser.add_argument(
        "--alerts",
        "-a",
        type=str,
        default=None,
        help="Path to save high-risk alerts JSON log (e.g. data/output/alerts.json)",
    )
    parser.add_argument(
        "--events",
        "-e",
        type=str,
        default=None,
        help="Path to save state transitions events JSON (e.g. data/output/events.json)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/authorization.yaml",
        help="Path to YAML configuration file with authorization and risk weights",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--tracker",
        "-t",
        type=str,
        default="bytetrack.yaml",
        help="Tracker configuration file (e.g. bytetrack.yaml)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="yolov8n.pt",
        help="YOLO model weights name or local path",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display video frames in a window during processing",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on: 'cpu', 'cuda', etc.",
    )
    return parser.parse_args()


def main() -> int:
    """Main CLI entrypoint."""
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"\n[ERROR] Input video file does not exist: {input_path.resolve()}", file=sys.stderr)
        print("Please provide a valid video file path.\n", file=sys.stderr)
        return 1

    # Load configuration
    config_file = Path(args.config)
    if config_file.exists():
        config = DetectionConfig.from_yaml(config_file)
    else:
        config = DetectionConfig()

    config.model_name = args.model
    config.tracker_type = args.tracker
    config.confidence_threshold = args.conf
    config.device = args.device

    print("\n" + "=" * 68)
    print(" VEHICLE THEFT DETECTION - PHASE 5 THEFT RISK SCORING ENGINE")
    print("=" * 68)
    print(f" Input Video:            {input_path}")
    print(f" Output Video:           {args.output if args.output else 'None (not saving)'}")
    print(f" Alerts Log:             {args.alerts if args.alerts else 'None (not saving)'}")
    print(f" Events Log:             {args.events if args.events else 'None (not saving)'}")
    print(f" Config File:            {config_file if config_file.exists() else 'Default Config'}")
    print(f" High Risk Threshold:    {config.high_risk_threshold}")
    print(f" Model:                  {args.model}")
    print(f" Tracker:                {args.tracker}")
    print(f" Confidence:             {args.conf}")
    print(f" Device:                 {args.device}")
    print(f" Show Display:           {args.show}")
    print("=" * 68 + "\n")

    try:
        print("[INFO] Initializing tracker, authorization, and risk engine...")
        tracker = ObjectTracker(config=config)
        parking_zone = ParkingZone(config.parking_zone_points)
        association_engine = AssociationEngine(config=config, parking_zone=parking_zone)
        behavior_engine = BehaviorEngine(config=config, parking_zone=parking_zone)

        # Load authorization store
        auth_store = (
            AuthorizationStore.from_yaml(config_file)
            if config_file.exists()
            else AuthorizationStore()
        )
        database = EventDatabase(config.db_path)
        risk_engine = RiskEngine(
            config=config,
            auth_store=auth_store,
            snapshots_dir=config.snapshots_dir,
            database=database,
        )

        processor = VideoProcessor(
            tracker=tracker,
            association_engine=association_engine,
            behavior_engine=behavior_engine,
            auth_store=auth_store,
            risk_engine=risk_engine,
            parking_zone=parking_zone,
            db=database,
            config=config,
        )

        print("[INFO] Processing video with multi-signal theft risk evaluation...")
        summary = processor.process_risk_assessment(
            input_path=input_path,
            output_path=args.output,
            alerts_output_path=args.alerts,
            events_output_path=args.events,
            show_display=args.show,
        )

        print("\n" + "-" * 68)
        print(" THEFT RISK EVALUATION COMPLETED SUCCESSFULLY")
        print("-" * 68)
        print(f" Frames Processed:              {summary['total_frames_processed']}")
        print(f" Unique Persons Observed:       {summary['unique_persons_count']}")
        print(f" Unique Vehicles Observed:      {summary['unique_vehicles_count']}")
        print(f" Peak Risk Score:               {summary['highest_risk_score']}")
        print(f" High-Risk Alerts Triggered:    {summary['total_high_risk_alerts']}")
        print(f" Elapsed Time:                  {summary['elapsed_seconds']}s")
        print(f" Average Speed:                 {summary['average_fps']} FPS")
        if summary["output_path"]:
            print(f" Saved Video:                   {summary['output_path']}")
        if summary["alerts_path"]:
            print(f" Saved Alerts Log:              {summary['alerts_path']}")
        if summary["events_path"]:
            print(f" Saved Events Log:              {summary['events_path']}")
        print("-" * 68 + "\n")
        return 0

    except Exception as exc:
        print(f"\n[ERROR] An error occurred during processing: {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
