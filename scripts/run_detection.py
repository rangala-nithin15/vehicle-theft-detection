"""CLI script to run YOLO object detection on a video file."""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DetectionConfig
from app.detector import YOLODetector
from app.video_processor import VideoProcessor


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run YOLO object detection on video to detect people and vehicles (Phase 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to input video file (e.g., data/input/test.mp4)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save annotated output video (e.g., data/output/detected.mp4)",
    )
    parser.add_argument(
        "--conf",
        "-c",
        type=float,
        default=0.35,
        help="Detection confidence threshold (0.0 to 1.0)",
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

    print("\n" + "=" * 60)
    print(" VEHICLE THEFT DETECTION - PHASE 1 OBJECT DETECTION")
    print("=" * 60)
    print(f" Input Video:   {input_path}")
    print(f" Output Video:  {args.output if args.output else 'None (not saving)'}")
    print(f" Model:         {args.model}")
    print(f" Confidence:    {args.conf}")
    print(f" Device:        {args.device}")
    print(f" Show Display:  {args.show}")
    print("=" * 60 + "\n")

    # Build custom configuration from arguments
    config = DetectionConfig(
        model_name=args.model,
        confidence_threshold=args.conf,
        device=args.device,
    )

    try:
        print("[INFO] Initializing YOLO detector...")
        detector = YOLODetector(config=config)
        processor = VideoProcessor(detector=detector, config=config)

        print("[INFO] Processing video...")
        summary = processor.process_video(
            input_path=input_path,
            output_path=args.output,
            show_display=args.show,
        )

        print("\n" + "-" * 60)
        print(" PROCESSING COMPLETED SUCCESSFULLY")
        print("-" * 60)
        print(f" Frames Processed: {summary['total_frames_processed']}")
        print(f" Total Detections: {summary['total_detections']}")
        print(f" Elapsed Time:     {summary['elapsed_seconds']}s")
        print(f" Average Speed:    {summary['average_fps']} FPS")
        if summary["output_path"]:
            print(f" Saved Output:     {summary['output_path']}")
        print("-" * 60 + "\n")
        return 0

    except Exception as exc:
        print(f"\n[ERROR] An error occurred during processing: {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
