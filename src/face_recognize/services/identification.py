"""Identification service for resolving face embeddings to names.

This module provides cache-aware identity resolution. It maintains
a per-track cache to avoid redundant database lookups.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..config import AppConfig
from ..core.models import BoundingBox, Face
from ..database.json_backend import JsonDatabase


@dataclass
class CachedIdentity:
    """Cached identification result for a track.

    Attributes:
        name: Resolved name ("Unknown" if not matched).
        confidence: Match confidence (0.0 if unknown).
        timestamp: Unix timestamp when cached.
        is_known: True if matched in database, False otherwise.
    """

    name: str
    confidence: float
    timestamp: float
    is_known: bool


@dataclass
class IdentifiedFace:
    """Face with resolved identity for rendering.

    Attributes:
        bbox: Bounding box coordinates for drawing.
        name: Person name or "Unknown".
        confidence: Match confidence (0.0 if unknown).
        is_known: True if matched in database.
        track_id: Tracking ID for reference.
    """

    bbox: BoundingBox
    name: str
    confidence: float
    is_known: bool
    track_id: int


class IdentificationService:
    """Service for identifying faces using database lookup with caching.

    Caching strategy:
    - Known identities: Never expire (cached until track is lost)
    - Unknown identities: Expire after `unknown_cache_ttl` seconds

    This prevents redundant database lookups while allowing
    unknown faces to be re-checked periodically.

    Attributes:
        database: JSON database backend.
        config: Application configuration.
    """

    def __init__(self, database: JsonDatabase, config: AppConfig) -> None:
        """Initialize the identification service.

        Args:
            database: Database backend for face lookups.
            config: Application configuration.
        """
        self.database = database
        self.config = config
        self._cache: dict[int, CachedIdentity] = {}

    def identify(self, faces: list[Face]) -> list[IdentifiedFace]:
        """Identify a list of faces.

        For each face:
        1. Check cache for existing identification
        2. If cache miss or expired, query database
        3. Update cache with result

        Args:
            faces: List of Face objects with track_id assigned.

        Returns:
            List of IdentifiedFace objects for rendering.

        Raises:
            ValueError: If any face has track_id=None.
        """
        results: list[IdentifiedFace] = []

        for face in faces:
            if face.track_id is None:
                raise ValueError(
                    "Face must have track_id assigned before identification"
                )

            identified = self._identify_single(face)
            results.append(identified)

        # Clean up cache entries for tracks no longer present
        active_tracks = {face.track_id for face in faces if face.track_id is not None}
        self._cleanup_cache(active_tracks)

        return results

    def _identify_single(self, face: Face) -> IdentifiedFace:
        """Identify a single face.

        Args:
            face: Face with track_id assigned.

        Returns:
            IdentifiedFace with resolved identity.
        """
        track_id = face.track_id
        assert track_id is not None  # Guaranteed by caller

        # Check cache
        cached = self._cache.get(track_id)
        if cached is not None and self._is_cache_valid(cached):
            return IdentifiedFace(
                bbox=face.bbox,
                name=cached.name,
                confidence=cached.confidence,
                is_known=cached.is_known,
                track_id=track_id,
            )

        # Cache miss or expired - query database
        match, score = self.database.search_by_embedding(
            face.embedding,
            self.config.similarity_threshold,
        )

        if match is not None:
            # Known person
            identity = CachedIdentity(
                name=match.name,
                confidence=score,
                timestamp=time.time(),
                is_known=True,
            )
        else:
            # Unknown person
            identity = CachedIdentity(
                name="Unknown",
                confidence=0.0,
                timestamp=time.time(),
                is_known=False,
            )

        # Update cache
        self._cache[track_id] = identity

        return IdentifiedFace(
            bbox=face.bbox,
            name=identity.name,
            confidence=identity.confidence,
            is_known=identity.is_known,
            track_id=track_id,
        )

    def _is_cache_valid(self, cached: CachedIdentity) -> bool:
        """Check if a cached identity is still valid.

        Known identities never expire.
        Unknown identities expire after unknown_cache_ttl seconds.

        Args:
            cached: Cached identity to check.

        Returns:
            True if cache entry is still valid.
        """
        if cached.is_known:
            # Known identities never expire
            return True

        # Unknown identities expire after TTL
        age = time.time() - cached.timestamp
        return age < self.config.unknown_cache_ttl

    def _cleanup_cache(self, active_tracks: set[int]) -> None:
        """Remove cache entries for tracks no longer active.

        Args:
            active_tracks: Set of currently active track IDs.
        """
        stale_tracks = set(self._cache.keys()) - active_tracks
        for track_id in stale_tracks:
            del self._cache[track_id]

    def clear_cache(self) -> None:
        """Clear all cached identifications."""
        self._cache.clear()

    def get_cache_size(self) -> int:
        """Get number of cached identifications.

        Returns:
            Number of entries in cache.
        """
        return len(self._cache)
