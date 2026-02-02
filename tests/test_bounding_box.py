"""Unit tests for BoundingBox class."""

import numpy as np

from src.face_recognize.core.models import BoundingBox


class TestBoundingBox:
    """Test cases for BoundingBox class."""

    def test_properties(self) -> None:
        """Test width, height, area, and center properties."""
        bbox = BoundingBox(x1=10, y1=20, x2=50, y2=80)

        assert bbox.width == 40
        assert bbox.height == 60
        assert bbox.area == 2400
        assert bbox.center == (30, 50)

    def test_to_tuple(self) -> None:
        """Test conversion to tuple."""
        bbox = BoundingBox(x1=10, y1=20, x2=50, y2=80)
        assert bbox.to_tuple() == (10, 20, 50, 80)

    def test_iou_no_overlap(self) -> None:
        """Test IoU calculation for non-overlapping boxes."""
        bbox1 = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        bbox2 = BoundingBox(x1=20, y1=20, x2=30, y2=30)

        assert bbox1.iou(bbox2) == 0.0
        assert bbox2.iou(bbox1) == 0.0

    def test_iou_partial_overlap(self) -> None:
        """Test IoU calculation for partially overlapping boxes."""
        bbox1 = BoundingBox(x1=0, y1=0, x2=10, y2=10)  # Area: 100
        bbox2 = BoundingBox(x1=5, y1=5, x2=15, y2=15)  # Area: 100

        # Intersection: (5,5) to (10,10) = 25
        # Union: 100 + 100 - 25 = 175
        # IoU: 25/175 = 0.142857...
        expected_iou = 25 / 175
        assert abs(bbox1.iou(bbox2) - expected_iou) < 0.001

    def test_iou_complete_overlap(self) -> None:
        """Test IoU calculation for identical boxes."""
        bbox1 = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        bbox2 = BoundingBox(x1=0, y1=0, x2=10, y2=10)

        assert bbox1.iou(bbox2) == 1.0

    def test_from_array(self) -> None:
        """Test creating BoundingBox from numpy array."""
        arr = np.array([10, 20, 50, 80])
        bbox = BoundingBox.from_array(arr)

        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 50
        assert bbox.y2 == 80
