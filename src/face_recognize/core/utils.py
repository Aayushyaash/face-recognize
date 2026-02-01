"""Utility functions for face detection and processing."""

from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt

from .models import BoundingBox


def compute_pairwise_iou(
    bboxes1: List[BoundingBox], bboxes2: List[BoundingBox]
) -> npt.NDArray[Any]:
    """Compute IoU matrix between two sets of bounding boxes.

    Args:
        bboxes1: First list of bounding boxes.
        bboxes2: Second list of bounding boxes.

    Returns:
        IoU matrix of shape (len(bboxes1), len(bboxes2)).
    """
    if not bboxes1 or not bboxes2:
        return np.zeros((len(bboxes1), len(bboxes2)))

    # Convert to numpy arrays for vectorized computation
    arr1 = np.array([[bbox.x1, bbox.y1, bbox.x2, bbox.y2] for bbox in bboxes1])
    arr2 = np.array([[bbox.x1, bbox.y1, bbox.x2, bbox.y2] for bbox in bboxes2])

    # Compute intersections
    inter_x1 = np.maximum(arr1[:, None, 0], arr2[:, 0])
    inter_y1 = np.maximum(arr1[:, None, 1], arr2[:, 1])
    inter_x2 = np.minimum(arr1[:, None, 2], arr2[:, 2])
    inter_y2 = np.minimum(arr1[:, None, 3], arr2[:, 3])

    # Calculate intersection areas
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_areas = inter_w * inter_h

    # Calculate areas of both sets of boxes
    areas1 = (arr1[:, 2] - arr1[:, 0]) * (arr1[:, 3] - arr1[:, 1])
    areas2 = (arr2[:, 2] - arr2[:, 0]) * (arr2[:, 3] - arr2[:, 1])

    # Calculate union areas and IoUs
    union_areas = areas1[:, None] + areas2 - inter_areas
    ious: npt.NDArray[Any] = inter_areas / np.maximum(union_areas, 1e-9)

    return ious


def find_best_matches(
    iou_matrix: npt.NDArray[Any], threshold: float = 0.3
) -> List[Tuple[int, int]]:
    """Find best matches between detections based on IoU threshold.

    Args:
        iou_matrix: IoU matrix of shape (num_detections1, num_detections2).
        threshold: Minimum IoU threshold for a match.

    Returns:
        List of tuples (idx1, idx2) representing matches.
    """
    matches = []

    # Create a copy of the matrix to mark used detections
    remaining_iou = iou_matrix.copy()

    while True:
        # Find the highest IoU
        max_idx = np.unravel_index(np.argmax(remaining_iou), remaining_iou.shape)
        max_iou = remaining_iou[max_idx]

        # If the max IoU is below threshold, stop
        if max_iou < threshold:
            break

        # Add the match
        matches.append((int(max_idx[0]), int(max_idx[1])))

        # Zero out the row and column to prevent reuse
        remaining_iou[max_idx[0], :] = 0
        remaining_iou[:, max_idx[1]] = 0

    return matches


def cosine_similarity(vec1: npt.NDArray[Any], vec2: npt.NDArray[Any]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Cosine similarity value between -1 and 1.
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))
