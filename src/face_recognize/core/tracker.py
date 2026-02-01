"""Face tracking module for persistent identity across frames.

This module provides face tracking using IoU-based bounding box matching
combined with embedding similarity verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ..config import AppConfig
from .models import BoundingBox, Face
from .utils import compute_pairwise_iou


@dataclass
class Track:
    """Internal state for a tracked face.

    Attributes:
        track_id: Unique persistent identifier for this track.
        bbox: Last known bounding box position.
        embedding: Last known face embedding (512-d normalized).
        age: Number of frames since last successful match.
        hits: Number of consecutive successful matches.
    """

    track_id: int
    bbox: BoundingBox
    embedding: npt.NDArray[np.float32]
    age: int = 0
    hits: int = 1


class FaceTracker:
    """Face tracker using IoU + embedding similarity matching.

    Maintains persistent track IDs across video frames. Uses a two-stage
    matching process:
    1. IoU-based spatial matching between bounding boxes
    2. Embedding similarity verification for matched pairs

    Attributes:
        config: Application configuration.
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialize the face tracker.

        Args:
            config: Application configuration with tracking parameters.
        """
        self.config = config
        self._tracks: list[Track] = []
        self._next_id: int = 1

    def update(self, faces: list[Face]) -> list[Face]:
        """Update tracks with new detections and assign track IDs.

        This is the main method to call each frame. It:
        1. Matches new faces to existing tracks using IoU
        2. Verifies matches using embedding similarity
        3. Creates new tracks for unmatched detections
        4. Removes stale tracks that haven't been matched

        Args:
            faces: List of detected Face objects (track_id will be None).

        Returns:
            List of Face objects with track_id assigned.
            The returned list is a new list with new Face objects.
        """
        if not faces:
            # Increment age of all tracks, remove stale ones
            self._age_tracks()
            return []

        if not self._tracks:
            # No existing tracks, create new ones for all faces
            return [self._create_tracked_face(face) for face in faces]

        # Stage 1: Compute IoU matrix between faces and tracks
        face_bboxes = [face.bbox for face in faces]
        track_bboxes = [track.bbox for track in self._tracks]
        iou_matrix = compute_pairwise_iou(face_bboxes, track_bboxes)

        # Stage 2: Find best matches using greedy algorithm
        matches, unmatched_faces, unmatched_tracks = self._match(iou_matrix, faces)

        # Stage 3: Update matched tracks and create output faces
        output_faces: list[Face] = []

        for face_idx, track_idx in matches:
            face = faces[face_idx]
            track = self._tracks[track_idx]

            # Update track state
            track.bbox = face.bbox
            track.embedding = face.embedding
            track.age = 0
            track.hits += 1

            # Create new Face with track_id
            output_faces.append(
                Face(
                    embedding=face.embedding,
                    bbox=face.bbox,
                    confidence=face.confidence,
                    landmarks=face.landmarks,
                    track_id=track.track_id,
                )
            )

        # Stage 4: Create new tracks for unmatched detections
        for face_idx in unmatched_faces:
            face = faces[face_idx]
            output_faces.append(self._create_tracked_face(face))

        # Stage 5: Age unmatched tracks
        for track_idx in unmatched_tracks:
            self._tracks[track_idx].age += 1

        # Stage 6: Remove stale tracks
        self._tracks = [
            track for track in self._tracks if track.age < self.config.max_track_age
        ]

        return output_faces

    def _match(
        self, iou_matrix: npt.NDArray[Any], faces: list[Face]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Match faces to tracks using IoU + embedding similarity.

        Uses greedy matching algorithm:
        1. Find highest IoU pair above threshold
        2. Verify with embedding similarity
        3. If verified, mark as match; otherwise mark both as unmatched
        4. Repeat until no more valid pairs

        Args:
            iou_matrix: IoU scores, shape (num_faces, num_tracks).
            faces: List of detected faces for embedding access.

        Returns:
            Tuple of:
            - matches: List of (face_idx, track_idx) pairs
            - unmatched_faces: List of face indices without matches
            - unmatched_tracks: List of track indices without matches
        """
        num_faces = len(faces)
        num_tracks = len(self._tracks)

        matches: list[tuple[int, int]] = []
        matched_faces: set[int] = set()
        matched_tracks: set[int] = set()

        # Create a working copy of IoU matrix
        working_iou = iou_matrix.copy()

        while True:
            # Find the highest IoU
            if working_iou.size == 0:
                break

            flat_idx = np.argmax(working_iou)
            face_idx, track_idx = np.unravel_index(flat_idx, working_iou.shape)
            max_iou = working_iou[face_idx, track_idx]

            # Stop if below IoU threshold
            if max_iou < self.config.iou_threshold:
                break

            # Verify with embedding similarity
            face_embedding = faces[face_idx].embedding
            track_embedding = self._tracks[track_idx].embedding
            similarity = float(np.dot(face_embedding, track_embedding))

            if similarity >= self.config.similarity_threshold:
                # Valid match
                matches.append((int(face_idx), int(track_idx)))
                matched_faces.add(int(face_idx))
                matched_tracks.add(int(track_idx))

            # Zero out this pair to prevent reuse
            working_iou[face_idx, :] = -1
            working_iou[:, track_idx] = -1

        # Determine unmatched
        unmatched_faces = [i for i in range(num_faces) if i not in matched_faces]
        unmatched_tracks = [i for i in range(num_tracks) if i not in matched_tracks]

        return matches, unmatched_faces, unmatched_tracks

    def _create_tracked_face(self, face: Face) -> Face:
        """Create a new track and return Face with track_id.

        Args:
            face: Detected face to create track for.

        Returns:
            New Face object with track_id assigned.
        """
        # Create new track
        track = Track(
            track_id=self._next_id,
            bbox=face.bbox,
            embedding=face.embedding,
        )
        self._tracks.append(track)
        self._next_id += 1

        # Return face with track_id
        return Face(
            embedding=face.embedding,
            bbox=face.bbox,
            confidence=face.confidence,
            landmarks=face.landmarks,
            track_id=track.track_id,
        )

    def _age_tracks(self) -> None:
        """Increment age of all tracks and remove stale ones."""
        for track in self._tracks:
            track.age += 1

        self._tracks = [
            track for track in self._tracks if track.age < self.config.max_track_age
        ]

    def reset(self) -> None:
        """Reset tracker state, clearing all tracks."""
        self._tracks = []
        self._next_id = 1

    def get_active_track_count(self) -> int:
        """Get number of currently active tracks.

        Returns:
            Number of tracks being maintained.
        """
        return len(self._tracks)
