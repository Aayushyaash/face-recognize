"""JSON file-based database backend for face storage.

This module provides a simple JSON file storage for face embeddings.
It supports CRUD operations and similarity-based search.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class PersonRecord:
    """A stored person record in the database.

    Attributes:
        id: Unique identifier (UUID4 string).
        name: Person's display name.
        embedding: 512-dimensional normalized face embedding.
        created_at: ISO 8601 timestamp of registration.
    """

    id: str
    name: str
    embedding: npt.NDArray[np.float32]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Convert record to JSON-serializable dictionary.

        Returns:
            Dictionary with id, name, embedding (as list), created_at.
        """
        return {
            "id": self.id,
            "name": self.name,
            "embedding": self.embedding.tolist(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonRecord:
        """Create PersonRecord from dictionary.

        Args:
            data: Dictionary with id, name, embedding, created_at keys.

        Returns:
            PersonRecord instance.
        """
        return cls(
            id=data["id"],
            name=data["name"],
            embedding=np.array(data["embedding"], dtype=np.float32),
            created_at=data["created_at"],
        )


class JsonDatabase:
    """JSON file-based database for face records.

    Thread-safe atomic writes using temp file + rename pattern.
    Auto-creates parent directories if they don't exist.

    Attributes:
        database_path: Path to the JSON database file.
    """

    def __init__(self, database_path: Path) -> None:
        """Initialize the database.

        Args:
            database_path: Path to the JSON file. Parent directories
                will be created if they don't exist.
        """
        self.database_path = database_path
        self._records: dict[str, PersonRecord] = {}
        self._load()

    def _load(self) -> None:
        """Load records from JSON file.

        If file doesn't exist, initializes empty database.
        If file is corrupted, raises ValueError.
        """
        if not self.database_path.exists():
            self._records = {}
            return

        try:
            with open(self.database_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate JSON structure
            if not isinstance(data, dict) or "persons" not in data:
                raise ValueError("Invalid database format: missing 'persons' key")

            self._records = {}
            for person_data in data["persons"]:
                record = PersonRecord.from_dict(person_data)
                self._records[record.name.lower()] = record

        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupted database file: {e}") from e

    def _save(self) -> None:
        """Save records to JSON file atomically.

        Uses temp file + rename pattern to prevent corruption.
        Creates parent directories if they don't exist.
        """
        # Ensure parent directory exists
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data
        data = {
            "version": "1.0",
            "persons": [record.to_dict() for record in self._records.values()],
        }

        # Write to temp file first, then rename (atomic on most filesystems)
        fd, temp_path = tempfile.mkstemp(
            suffix=".json",
            dir=self.database_path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Atomic rename
            os.replace(temp_path, self.database_path)
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def add(self, name: str, embedding: npt.NDArray[np.float32]) -> PersonRecord:
        """Add a new person to the database.

        Args:
            name: Person's name (case-insensitive for duplicate check).
            embedding: 512-dimensional normalized face embedding.

        Returns:
            The created PersonRecord.

        Raises:
            ValueError: If name already exists in database.
            ValueError: If embedding is not 512-dimensional.
        """
        # Validate embedding shape
        if embedding.shape != (512,):
            raise ValueError(
                f"Embedding must be 512-dimensional, got {embedding.shape}"
            )

        # Check for duplicate name (case-insensitive)
        name_key = name.lower()
        if name_key in self._records:
            raise ValueError(f"Person '{name}' already exists in database")

        # Create new record
        record = PersonRecord(
            id=str(uuid.uuid4()),
            name=name,
            embedding=embedding.copy(),
            created_at=datetime.now().isoformat(),
        )

        self._records[name_key] = record
        self._save()

        return record

    def delete(self, name: str) -> bool:
        """Delete a person from the database.

        Args:
            name: Person's name (case-insensitive).

        Returns:
            True if person was deleted, False if not found.
        """
        name_key = name.lower()
        if name_key not in self._records:
            return False

        del self._records[name_key]
        self._save()
        return True

    def get(self, name: str) -> PersonRecord | None:
        """Get a person by name.

        Args:
            name: Person's name (case-insensitive).

        Returns:
            PersonRecord if found, None otherwise.
        """
        return self._records.get(name.lower())

    def list_all(self) -> list[PersonRecord]:
        """List all registered persons.

        Returns:
            List of all PersonRecord objects, sorted by name.
        """
        return sorted(self._records.values(), key=lambda r: r.name.lower())

    def count(self) -> int:
        """Get the number of registered persons.

        Returns:
            Number of persons in database.
        """
        return len(self._records)

    def search_by_embedding(
        self, embedding: npt.NDArray[np.float32], threshold: float
    ) -> tuple[PersonRecord | None, float]:
        """Search for the best matching person by embedding similarity.

        Uses cosine similarity (dot product of normalized vectors).

        Args:
            embedding: 512-dimensional normalized query embedding.
            threshold: Minimum similarity score to consider a match.

        Returns:
            Tuple of (best_match, similarity_score).
            Returns (None, 0.0) if no match above threshold.
        """
        if not self._records:
            return None, 0.0

        best_match: PersonRecord | None = None
        best_score: float = 0.0

        for record in self._records.values():
            # Cosine similarity = dot product of normalized vectors
            score = float(np.dot(embedding, record.embedding))

            if score > best_score:
                best_score = score
                best_match = record

        if best_score >= threshold:
            return best_match, best_score

        return None, 0.0
