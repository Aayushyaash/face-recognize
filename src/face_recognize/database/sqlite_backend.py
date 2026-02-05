"""Encrypted SQLite database backend for face recognition system.

This module implements a SQLite-based database backend that stores
face embeddings and associated metadata with field-level encryption,
providing CRUD operations and similarity search capabilities.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..database.base import PersonRecord
from .encryption import FieldEncryptor


class EncryptedSqliteDatabase:
    """Encrypted SQLite database backend for storing face embeddings and metadata.

    This backend provides thread-safe CRUD operations with field-level encryption
    and similarity search using cosine similarity for face recognition.
    """

    def __init__(
        self, db_path: str | Path, encryption_key: bytes | None = None
    ) -> None:
        """Initialize the encrypted SQLite database.

        Args:
            db_path: Path to the SQLite database file.
            encryption_key: Encryption key to use. If None, generates a new key.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.encryptor = FieldEncryptor(encryption_key)

        # Initialize the database schema
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Initialize the database schema if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create persons table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,  -- Encrypted
                    embedding TEXT NOT NULL,  -- Encrypted
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Create index on name for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_persons_name ON persons (name)
            """)

            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper configuration.

        Returns:
            Configured SQLite connection.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn

    def add(self, name: str, embedding: npt.NDArray[np.float32]) -> PersonRecord:
        """Add a new person to the database.

        Args:
            name: Person's name.
            embedding: 512-dimensional face embedding vector.

        Returns:
            Created PersonRecord with assigned ID and timestamps.

        Raises:
            ValueError: If name already exists in database.
        """
        person_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        # Encrypt the name and embedding
        encrypted_name = self.encryptor.encrypt_field(name)
        encrypted_embedding = self.encryptor.encrypt_embedding(embedding)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            try:
                cursor.execute(
                    """
                    INSERT INTO persons (id, name, embedding, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        person_id,
                        encrypted_name,
                        encrypted_embedding,
                        timestamp,
                        timestamp,
                    ),
                )

                conn.commit()
            except sqlite3.IntegrityError as e:
                if "name" in str(e):
                    raise ValueError(f"Person with name '{name}' already exists")
                raise e

        # Return the created record
        return PersonRecord(
            id=person_id,
            name=name,  # Return unencrypted name
            embedding=embedding,  # Return unencrypted embedding
            created_at=timestamp,
        )

    def get(self, name: str) -> PersonRecord | None:
        """Get a person by name.

        Args:
            name: Person's name to look up.

        Returns:
            PersonRecord if found, None otherwise.
        """
        # Encrypt the name for comparison
        encrypted_name = self.encryptor.encrypt_field(name)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM persons WHERE name = ?", (encrypted_name,))
            row = cursor.fetchone()

            if row is None:
                return None

            # Decrypt the name and embedding
            decrypted_name = self.encryptor.decrypt_field(row["name"])
            decrypted_embedding = self.encryptor.decrypt_embedding(row["embedding"])

            return PersonRecord(
                id=row["id"],
                name=decrypted_name,
                embedding=decrypted_embedding,
                created_at=row["created_at"],
            )

    def update(
        self,
        name: str,
        new_name: str | None = None,
        embedding: npt.NDArray[np.float32] | None = None,
    ) -> bool:
        """Update a person's information.

        Args:
            name: Current name of the person to update.
            new_name: New name (optional).
            embedding: New embedding (optional).

        Returns:
            True if update was successful, False if person not found.
        """
        # Encrypt the current name for lookup
        encrypted_current_name = self.encryptor.encrypt_field(name)

        # Prepare updates
        updates = []
        params = []

        if new_name is not None:
            encrypted_new_name = self.encryptor.encrypt_field(new_name)
            updates.append("name = ?")
            params.append(encrypted_new_name)
        if embedding is not None:
            encrypted_embedding = self.encryptor.encrypt_embedding(embedding)
            updates.append("embedding = ?")
            params.append(encrypted_embedding)

        if not updates:
            # Nothing to update, just check if person exists
            return self.get(name) is not None

        # Add updated_at timestamp
        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(encrypted_current_name)  # For WHERE clause

        with self._get_connection() as conn:
            cursor = conn.cursor()

            try:
                # Build the SQL query with proper parameterization
                # Constructing SET clause dynamically but values are parameterized
                updates_clause = ", ".join(updates)
                # Construct the query string - only column names are dynamic
                # and they are hardcoded in the calling code
                sql_query = f"UPDATE persons SET {updates_clause} WHERE name = ?"
                cursor.execute(sql_query, params)

                conn.commit()

                # Return True if any rows were affected
                return cursor.rowcount > 0
            except sqlite3.IntegrityError as e:
                if "name" in str(e):
                    raise ValueError(f"Person with name '{new_name}' already exists")
                raise e

    def delete(self, name: str) -> bool:
        """Delete a person from the database.

        Args:
            name: Person's name to delete.

        Returns:
            True if deletion was successful, False if person not found.
        """
        # Encrypt the name for lookup
        encrypted_name = self.encryptor.encrypt_field(name)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM persons WHERE name = ?", (encrypted_name,))
            conn.commit()

            return cursor.rowcount > 0

    def list_all(self) -> list[PersonRecord]:
        """List all persons in the database.

        Returns:
            List of all PersonRecord objects.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM persons ORDER BY name")
            rows = cursor.fetchall()

            records = []
            for row in rows:
                # Decrypt the name and embedding
                decrypted_name = self.encryptor.decrypt_field(row["name"])
                decrypted_embedding = self.encryptor.decrypt_embedding(row["embedding"])

                records.append(
                    PersonRecord(
                        id=row["id"],
                        name=decrypted_name,
                        embedding=decrypted_embedding,
                        created_at=row["created_at"],
                    )
                )

            return records

    def count(self) -> int:
        """Count the number of persons in the database.

        Returns:
            Number of persons in the database.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM persons")
            result = cursor.fetchone()[0]
            return int(result)

    def search_by_embedding(
        self, embedding: npt.NDArray[np.float32], threshold: float = 0.4
    ) -> tuple[PersonRecord | None, float]:
        """Search for the most similar person to the given embedding.

        Args:
            embedding: Face embedding to search for.
            threshold: Minimum similarity score to consider a match.

        Returns:
            Tuple of (PersonRecord, similarity_score) if match found and above
            threshold,
            otherwise (None, 0.0).
        """
        # Get all records from the database
        all_records = self.list_all()

        best_match: PersonRecord | None = None
        best_score = 0.0

        for record in all_records:
            # Calculate cosine similarity
            similarity = float(np.dot(embedding, record.embedding))

            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = record

        if best_match is None:
            return None, 0.0

        return best_match, best_score

    def clear(self) -> None:
        """Clear all persons from the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM persons")
            conn.commit()
