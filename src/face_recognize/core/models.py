"""Core data models for face detection and tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BoundingBox:
    """Face bounding box coordinates.

    Attributes:
        x1: Left coordinate.
        y1: Top coordinate.
        x2: Right coordinate.
        y2: Bottom coordinate.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """Bounding box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Bounding box height."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Bounding box area in pixels."""
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        """Center point (x, y) of bounding box."""
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Return coordinates as (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)

    def iou(self, other: BoundingBox) -> float:
        """Calculate Intersection over Union with another box.

        Args:
            other: Another BoundingBox to compare.

        Returns:
            IoU score between 0 and 1.
        """
        # Calculate intersection
        xi1 = max(self.x1, other.x1)
        yi1 = max(self.y1, other.y1)
        xi2 = min(self.x2, other.x2)
        yi2 = min(self.y2, other.y2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0

    @classmethod
    def from_array(cls, arr: np.ndarray) -> BoundingBox:
        """Create BoundingBox from numpy array [x1, y1, x2, y2].

        Args:
            arr: Numpy array with 4 coordinates.

        Returns:
            BoundingBox instance.
        """
        return cls(
            x1=int(arr[0]),
            y1=int(arr[1]),
            x2=int(arr[2]),
            y2=int(arr[3]),
        )


@dataclass
class Face:
    """Detected face with embedding and metadata.

    Attributes:
        embedding: 512-dimensional normalized face embedding.
        bbox: Bounding box coordinates.
        confidence: Detection confidence score (0-1).
        landmarks: 5-point facial landmarks array (5, 2).
        track_id: Persistent tracking ID (assigned by tracker).
    """

    embedding: np.ndarray
    bbox: BoundingBox
    confidence: float
    landmarks: np.ndarray
    track_id: Optional[int] = None

    def similarity(self, other: Face) -> float:
        """Calculate cosine similarity with another face.

        Args:
            other: Another Face to compare.

        Returns:
            Similarity score between -1 and 1.
        """
        return float(np.dot(self.embedding, other.embedding))

    def similarity_to_embedding(self, embedding: np.ndarray) -> float:
        """Calculate cosine similarity with an embedding vector.

        Args:
            embedding: 512-d normalized embedding vector.

        Returns:
            Similarity score between -1 and 1.
        """
        return float(np.dot(self.embedding, embedding))