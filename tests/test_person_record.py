"""Unit tests for the PersonRecord dataclass in the database module."""

import numpy as np
import pytest

from src.face_recognize.database.json_backend import PersonRecord


def test_person_record_creation():
    """Test creating a PersonRecord instance."""
    embedding = np.random.rand(512).astype(np.float32)
    record = PersonRecord(
        id="test-id-123",
        name="John Doe",
        embedding=embedding,
        created_at="2026-01-30T10:15:23.456789"
    )
    
    assert record.id == "test-id-123"
    assert record.name == "John Doe"
    assert record.created_at == "2026-01-30T10:15:23.456789"
    assert record.embedding.shape == (512,)
    assert record.embedding.dtype == np.float32


def test_person_record_to_dict():
    """Test converting PersonRecord to dictionary."""
    embedding = np.random.rand(512).astype(np.float32)
    record = PersonRecord(
        id="test-id-123",
        name="John Doe",
        embedding=embedding,
        created_at="2026-01-30T10:15:23.456789"
    )
    
    record_dict = record.to_dict()
    
    assert record_dict["id"] == "test-id-123"
    assert record_dict["name"] == "John Doe"
    assert record_dict["created_at"] == "2026-01-30T10:15:23.456789"
    assert isinstance(record_dict["embedding"], list)
    assert len(record_dict["embedding"]) == 512


def test_person_record_from_dict():
    """Test creating PersonRecord from dictionary."""
    embedding_list = np.random.rand(512).astype(np.float32).tolist()
    data = {
        "id": "test-id-123",
        "name": "John Doe",
        "embedding": embedding_list,
        "created_at": "2026-01-30T10:15:23.456789"
    }
    
    record = PersonRecord.from_dict(data)
    
    assert record.id == "test-id-123"
    assert record.name == "John Doe"
    assert record.created_at == "2026-01-30T10:15:23.456789"
    assert record.embedding.shape == (512,)
    assert record.embedding.dtype == np.float32
    assert np.array_equal(record.embedding, np.array(embedding_list, dtype=np.float32))


def test_person_record_embedding_not_auto_copied():
    """Test that embedding is not automatically copied when creating PersonRecord directly."""
    original_embedding = np.random.rand(512).astype(np.float32)
    record = PersonRecord(
        id="test-id-123",
        name="John Doe",
        embedding=original_embedding,
        created_at="2026-01-30T10:15:23.456789"
    )

    # Modify original embedding - this will affect the record's embedding too
    # since PersonRecord stores a reference, not a copy
    original_embedding.fill(0.0)

    # The record's embedding will be affected since it shares the same reference
    assert np.allclose(record.embedding, 0.0)