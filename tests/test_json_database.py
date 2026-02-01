"""Unit tests for the JsonDatabase class in the database module."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.face_recognize.database.json_backend import JsonDatabase


def test_json_database_initialization() -> None:
    """Test initializing a JsonDatabase instance."""
    # Create a temporary file path, but don't create the file yet
    db_path = Path(tempfile.mktemp())

    try:
        db = JsonDatabase(db_path)

        assert db.database_path == db_path
        assert len(db.list_all()) == 0  # Should be empty initially
    finally:
        # Clean up
        if db_path.exists():
            os.unlink(db_path)


def test_json_database_add_person() -> None:
    """Test adding a person to the database."""
    # Create a temporary file path, but don't create the file yet
    db_path = Path(tempfile.mktemp())

    try:
        db = JsonDatabase(db_path)
        embedding = np.random.rand(512).astype(np.float32)

        record = db.add("John Doe", embedding)

        assert record.name == "John Doe"
        assert record.embedding.shape == (512,)
        assert np.array_equal(record.embedding, embedding)
        assert len(db.list_all()) == 1

        # Verify the record can be retrieved
        retrieved = db.get("John Doe")
        assert retrieved is not None
        assert retrieved.name == "John Doe"
        assert np.array_equal(retrieved.embedding, embedding)
    finally:
        # Clean up
        if db_path.exists():
            os.unlink(db_path)


def test_json_database_duplicate_name() -> None:
    """Test that adding a person with duplicate name raises an error."""
    # Create a temporary file path, but don't create the file yet
    db_path = Path(tempfile.mktemp())

    try:
        db = JsonDatabase(db_path)
        embedding1 = np.random.rand(512).astype(np.float32)
        embedding2 = np.random.rand(512).astype(np.float32)

        # Add first person
        _ = db.add("John Doe", embedding1)

        # Attempt to add another person with same name (case-insensitive)
        with pytest.raises(
            ValueError, match=r"Person 'john doe' already exists in database"
        ):
            db.add("john doe", embedding2)  # Lowercase should also conflict
    finally:
        # Clean up
        if db_path.exists():
            os.unlink(db_path)


def test_json_database_invalid_embedding_shape() -> None:
    """Test that adding a person with invalid embedding shape raises an error."""
    # Create a temporary file path, but don't create the file yet
    db_path = Path(tempfile.mktemp())

    try:
        db = JsonDatabase(db_path)
        invalid_embedding = np.random.rand(256).astype(np.float32)  # Wrong size

        with pytest.raises(ValueError, match=r"Embedding must be 512-dimensional"):
            db.add("John Doe", invalid_embedding)
    finally:
        # Clean up
        if db_path.exists():
            os.unlink(db_path)


def test_json_database_get_person() -> None:
    """Test retrieving a person from the database."""
    # Create a temporary file path, but don't create the file yet
    db_path = Path(tempfile.mktemp())

    try:
        db = JsonDatabase(db_path)
        embedding = np.random.rand(512).astype(np.float32)

        # Add a person
        _ = db.add("Jane Smith", embedding)

        # Retrieve the person (case-insensitive)
        retrieved_record = db.get("jane smith")
        assert retrieved_record is not None
        assert retrieved_record.name == "Jane Smith"
        assert np.array_equal(retrieved_record.embedding, embedding)

        # Try to get a non-existent person
        nonexistent = db.get("Non Existent")
        assert nonexistent is None
    finally:
        # Clean up
        if db_path.exists():
            os.unlink(db_path)


def test_json_database_delete_person() -> None:
    """Test deleting a person from the database."""
    # Create a temporary file path, but don't create the file yet
    db_path = Path(tempfile.mktemp())

    try:
        db = JsonDatabase(db_path)
        embedding = np.random.rand(512).astype(np.float32)

        # Add a person
        db.add("Jane Smith", embedding)
        assert len(db.list_all()) == 1

        # Delete the person (case-insensitive)
        result = db.delete("jane smith")
        assert result is True
        assert len(db.list_all()) == 0

        # Try to delete a non-existent person
        result = db.delete("Non Existent")
        assert result is False
    finally:
        # Clean up
        if db_path.exists():
            os.unlink(db_path)


def test_json_database_list_all_and_count() -> None:
    """Test listing all persons and counting them."""
    # Create a temporary file path, but don't create the file yet
    db_path = Path(tempfile.mktemp())

    try:
        db = JsonDatabase(db_path)
        embedding1 = np.random.rand(512).astype(np.float32)
        embedding2 = np.random.rand(512).astype(np.float32)

        # Add two people
        db.add("Jane Smith", embedding1)
        db.add("John Doe", embedding2)

        # Check count
        assert db.count() == 2

        # Check list
        all_records = db.list_all()
        assert len(all_records) == 2

        # Names should be sorted alphabetically (case-insensitive)
        names = [record.name for record in all_records]
        assert "Jane Smith" in names
        assert "John Doe" in names
        assert names == sorted(names, key=str.lower)
    finally:
        # Clean up
        if db_path.exists():
            os.unlink(db_path)


def test_json_database_save_and_load() -> None:
    """Test saving to and loading from JSON file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_db.json"

        # Create and populate database
        db1 = JsonDatabase(db_path)
        embedding1 = np.random.rand(512).astype(np.float32)
        embedding2 = np.random.rand(512).astype(np.float32)

        _ = db1.add("Alice Johnson", embedding1)
        _ = db1.add("Bob Wilson", embedding2)

        # Close and reopen database
        del db1
        db2 = JsonDatabase(db_path)

        # Verify data was saved and loaded correctly
        assert db2.count() == 2

        alice = db2.get("Alice Johnson")
        assert alice is not None
        assert alice.name == "Alice Johnson"
        assert np.array_equal(alice.embedding, embedding1)

        bob = db2.get("Bob Wilson")
        assert bob is not None
        assert bob.name == "Bob Wilson"
        assert np.array_equal(bob.embedding, embedding2)


def test_json_database_search_by_embedding() -> None:
    """Test searching for a person by embedding similarity."""
    # Create a temporary file path, but don't create the file yet
    db_path = Path(tempfile.mktemp())

    try:
        db = JsonDatabase(db_path)
        embedding1 = np.random.rand(512).astype(np.float32)
        embedding1 /= np.linalg.norm(embedding1)  # Normalize
        embedding2 = np.random.rand(512).astype(np.float32)
        embedding2 /= np.linalg.norm(embedding2)  # Normalize

        # Add two people to the database
        _ = db.add("Alice Johnson", embedding1)
        _ = db.add("Bob Wilson", embedding2)

        # Search for Alice with her own embedding (should find exact match)
        match, score = db.search_by_embedding(embedding1, 0.5)
        assert match is not None
        assert match.name == "Alice Johnson"
        assert score > 0.99  # Very high similarity

        # Search with an unknown embedding (should not find match if threshold is high)
        unknown_embedding = np.random.rand(512).astype(np.float32)
        unknown_embedding /= np.linalg.norm(unknown_embedding)  # Normalize
        match, score = db.search_by_embedding(unknown_embedding, 0.9)  # High threshold
        assert match is None
        assert score == 0.0
    finally:
        # Clean up
        if db_path.exists():
            os.unlink(db_path)


def test_json_database_empty_search() -> None:
    """Test searching in an empty database."""
    # Create a temporary file path, but don't create the file yet
    db_path = Path(tempfile.mktemp())

    try:
        db = JsonDatabase(db_path)
        unknown_embedding = np.random.rand(512).astype(np.float32)

        # Search in empty database
        match, score = db.search_by_embedding(unknown_embedding, 0.5)
        assert match is None
        assert score == 0.0
    finally:
        # Clean up
        if db_path.exists():
            os.unlink(db_path)
