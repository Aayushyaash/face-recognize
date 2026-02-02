"""Unit tests for utility functions."""

import numpy as np

from src.face_recognize.core.models import BoundingBox
from src.face_recognize.core.utils import (
    compute_pairwise_iou,
    cosine_similarity,
    find_best_matches,
)


class TestUtils:
    """Test cases for utility functions."""

    def test_compute_pairwise_iou_empty_lists(self) -> None:
        """Test IoU computation with empty lists."""
        iou_matrix = compute_pairwise_iou([], [])
        assert iou_matrix.shape == (0, 0)

        iou_matrix = compute_pairwise_iou([BoundingBox(0, 0, 10, 10)], [])
        assert iou_matrix.shape == (1, 0)

        iou_matrix = compute_pairwise_iou([], [BoundingBox(0, 0, 10, 10)])
        assert iou_matrix.shape == (0, 1)

    def test_compute_pairwise_iou_non_overlapping(self) -> None:
        """Test IoU computation for non-overlapping boxes."""
        bbox1 = BoundingBox(0, 0, 10, 10)
        bbox2 = BoundingBox(20, 20, 30, 30)

        iou_matrix = compute_pairwise_iou([bbox1], [bbox2])
        assert iou_matrix.shape == (1, 1)
        assert iou_matrix[0, 0] == 0.0

    def test_compute_pairwise_iou_overlapping(self) -> None:
        """Test IoU computation for overlapping boxes."""
        # bbox1: area 100 (0,0,10,10)
        # bbox2: area 100 (5,5,15,15)
        # intersection: area 25 (5,5,10,10)
        # union: 100 + 100 - 25 = 175
        # iou: 25/175 = 0.142857...
        bbox1 = BoundingBox(0, 0, 10, 10)
        bbox2 = BoundingBox(5, 5, 15, 15)

        iou_matrix = compute_pairwise_iou([bbox1], [bbox2])
        assert iou_matrix.shape == (1, 1)
        expected_iou = 25 / 175
        assert abs(iou_matrix[0, 0] - expected_iou) < 0.001

    def test_find_best_matches_no_matches(self) -> None:
        """Test finding matches when no IoU exceeds threshold."""
        iou_matrix = np.array([[0.1, 0.2], [0.2, 0.1]])  # All below threshold 0.3
        matches = find_best_matches(iou_matrix, threshold=0.3)
        assert len(matches) == 0

    def test_find_best_matches_with_matches(self) -> None:
        """Test finding matches when some IoUs exceed threshold."""
        # Matrix: [[0.1, 0.8], [0.2, 0.4]]
        # With threshold 0.3: (0,1) with 0.8 and (1,1) with 0.4 are candidates
        # But (0,1) is better, so (1,1) won't be matched due to greedy approach
        iou_matrix = np.array([[0.1, 0.8], [0.2, 0.4]])
        matches = find_best_matches(iou_matrix, threshold=0.3)

        # Greedy: match (0,1) first, zero col 1
        # Remaining row 1 col 0 is 0.2 < 0.3, so only one match
        assert len(matches) == 1
        assert (0, 1) in matches

    def test_cosine_similarity_perpendicular_vectors(self) -> None:
        """Test cosine similarity for perpendicular vectors."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])

        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 0.001

    def test_cosine_similarity_parallel_vectors(self) -> None:
        """Test cosine similarity for parallel vectors."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([2.0, 0.0])  # Same direction, different magnitude

        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.001

    def test_cosine_similarity_opposite_vectors(self) -> None:
        """Test cosine similarity for opposite vectors."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([-1.0, 0.0])  # Opposite direction

        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 0.001

    def test_cosine_similarity_with_zero_vector(self) -> None:
        """Test cosine similarity when one vector is zero."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 0.0])

        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 0.001
