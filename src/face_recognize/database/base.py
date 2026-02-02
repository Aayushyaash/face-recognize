"""Database abstraction layer.

This module defines the protocol that any database backend must implement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

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


class DatabaseBackend(Protocol):
    """Protocol for database backends."""

    def add(self, name: str, embedding: npt.NDArray[np.float32]) -> PersonRecord:
        """Add a new person to the database.

        Args:
            name: Person's name.
            embedding: 512-dimensional normalized face embedding.

        Returns:
            The created PersonRecord.

        Raises:
            ValueError: If name already exists or embedding is invalid.
        """
        ...

    def get(self, name: str) -> PersonRecord | None:
        """Get a person by name.

        Args:
            name: Person's name.

        Returns:
            PersonRecord if found, None otherwise.
        """
        ...

    def delete(self, name: str) -> bool:
        """Delete a person from the database.

        Args:
            name: Person's name.

        Returns:
            True if person was deleted, False if not found.
        """
        ...

    def list_all(self) -> list[PersonRecord]:
        """List all registered persons.

        Returns:
            List of all PersonRecord objects.
        """
        ...

    def count(self) -> int:
        """Get the number of registered persons.

        Returns:
            Number of persons in database.
        """
        ...

    def search_by_embedding(
        self, embedding: npt.NDArray[np.float32], threshold: float
    ) -> tuple[PersonRecord | None, float]:
        """Search for the best matching person by embedding similarity.

        Args:
            embedding: 512-dimensional normalized query embedding.
            threshold: Minimum similarity score to consider a match.

        Returns:
            Tuple of (best_match, similarity_score).
            Returns (None, 0.0) if no match above threshold.
        """
        ...
