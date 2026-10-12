"""Spatial zones module for parking area polygons and vehicle interaction zones."""

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


class ParkingZone:
    """Represents a monitored polygon region representing the parking area."""

    def __init__(
        self,
        points: Sequence[Sequence[int]],
        name: str = "PARKING AREA",
    ) -> None:
        """Initialize the parking zone with polygon vertices.

        Args:
            points: Sequence of [x, y] coordinates forming the polygon.
            name: Human-readable name for the zone.
        """
        if len(points) < 3:
            raise ValueError(
                "ParkingZone polygon must have at least 3 points."
            )

        self.name = name
        self.raw_points = [list(pt) for pt in points]

        self.polygon = np.array(
            self.raw_points,
            dtype=np.int32,
        ).reshape((-1, 1, 2))

    def contains_point(
        self,
        point: Tuple[int, int],
    ) -> bool:
        """Check whether a point is inside or on the parking-zone boundary.

        Args:
            point: Tuple of (x, y) coordinates.

        Returns:
            True if the point is inside or on the boundary.
            False otherwise.
        """
        pt = (
            float(point[0]),
            float(point[1]),
        )

        result = cv2.pointPolygonTest(
            self.polygon,
            pt,
            measureDist=False,
        )

        return result >= 0

    def contains_box(
        self,
        bbox: Tuple[int, int, int, int],
        check_bottom_center: bool = True,
    ) -> bool:
        """Check whether a bounding box is inside the parking zone.

        Args:
            bbox: Bounding box as (x1, y1, x2, y2).
            check_bottom_center:
                If True, use the bottom-center point.
                If False, use the geometric center.

        Returns:
            True if the selected reference point is inside
            the parking zone.
        """
        x1, y1, x2, y2 = bbox

        center_x = (x1 + x2) // 2

        if check_bottom_center:
            test_point = (
                center_x,
                y2,
            )
        else:
            center_y = (y1 + y2) // 2
            test_point = (
                center_x,
                center_y,
            )

        return self.contains_point(test_point)

    def draw(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 220, 0),
        thickness: int = 2,
        fill_alpha: float = 0.10,
    ) -> np.ndarray:
        """Return the frame without displaying the parking zone.

        The parking-zone polygon is still stored and used internally
        by contains_point() and contains_box().

        The following are intentionally NOT displayed:

        - Green parking-area fill
        - Green parking-area boundary
        - [PARKING AREA] label

        The parameters are kept for compatibility with existing code.
        """
        return frame.copy()


def get_vehicle_interaction_zone(
    bbox: Tuple[int, int, int, int],
    margin_px: int = 50,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """Calculate an expanded interaction area around a vehicle.

    Args:
        bbox:
            Vehicle bounding box as (x1, y1, x2, y2).

        margin_px:
            Number of pixels to expand around the vehicle.

        frame_width:
            Optional frame width used to clamp the result.

        frame_height:
            Optional frame height used to clamp the result.

    Returns:
        Expanded rectangle as:
        (ix1, iy1, ix2, iy2)
    """
    x1, y1, x2, y2 = bbox

    ix1 = max(
        0,
        x1 - margin_px,
    )

    iy1 = max(
        0,
        y1 - margin_px,
    )

    ix2 = x2 + margin_px
    iy2 = y2 + margin_px

    if frame_width is not None:
        ix2 = min(
            frame_width - 1,
            ix2,
        )

    if frame_height is not None:
        iy2 = min(
            frame_height - 1,
            iy2,
        )

    return (
        ix1,
        iy1,
        ix2,
        iy2,
    )


def is_point_in_rect(
    point: Tuple[int, int],
    rect: Tuple[int, int, int, int],
) -> bool:
    """Check whether a point is inside a rectangle.

    Args:
        point:
            Point as (x, y).

        rect:
            Rectangle as (x1, y1, x2, y2).

    Returns:
        True if the point is inside the rectangle.
        False otherwise.
    """
    px, py = point

    rx1, ry1, rx2, ry2 = rect

    return (
        rx1 <= px <= rx2
        and ry1 <= py <= ry2
    )