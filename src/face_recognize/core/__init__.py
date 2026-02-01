"""Core face detection and tracking modules."""

from .detector import FaceDetector
from .models import BoundingBox, Face
from .tracker import FaceTracker, Track
from .utils import compute_pairwise_iou, cosine_similarity, find_best_matches

__all__ = [
    "BoundingBox",
    "Face",
    "FaceDetector",
    "FaceTracker",
    "Track",
    "compute_pairwise_iou",
    "cosine_similarity",
    "find_best_matches",
]
