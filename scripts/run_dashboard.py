"""Launcher script for the Vehicle Theft Detection Streamlit Dashboard.

Usage:
    python scripts/run_dashboard.py
"""

import os
from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
DASHBOARD_PATH = BASE_DIR / "app" / "dashboard.py"


def main() -> None:
    """Launch the Streamlit dashboard application."""
    print("=" * 70)
    print("  LAUNCHING VEHICLE THEFT DETECTION SURVEILLANCE DASHBOARD")
    print("=" * 70)
    print(f"Target file: {DASHBOARD_PATH}")
    print("Opening Streamlit web application on local network...\n")

    cmd = [sys.executable, "-m", "streamlit", "run", str(DASHBOARD_PATH)]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nSurveillance dashboard terminated by user.")
    except Exception as exc:
        print(f"\nFailed to launch dashboard: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
