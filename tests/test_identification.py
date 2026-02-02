"""Unit tests for IdentificationService class."""

import time
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

from src.face_recognize.config import AppConfig
from src.face_recognize.core.models import BoundingBox, Face
from src.face_recognize.database.json_backend import JsonDatabase
from src.face_recognize.services.identification import (
    CachedIdentity,
    IdentificationService,
)


@pytest.fixture
def mock_db() -> mock.Mock:
    """Mocked JsonDatabase with search_by_embedding method."""
    db_mock = mock.Mock(spec=JsonDatabase)
    return db_mock


@pytest.fixture
def service(mock_db: mock.Mock) -> IdentificationService:
    """IdentificationService instance using mock_db."""
    config = AppConfig(
        model="test_model.onnx",
        camera_index=0,
        frame_width=640,
        frame_height=480,
        iou_threshold=0.3,
        similarity_threshold=0.7,
        max_track_age=30,
        unknown_cache_ttl=300,
        min_quality_score=0.5,
        box_thickness=2,
        font_scale=0.7,
        known_color=(0, 255, 0),
        unknown_color=(0, 0, 255),
        text_color=(255, 255, 255),
        database_path=Path("data/database.json"),
    )
    return IdentificationService(mock_db, config)


def test_identify_cache_hit_known(
    service: IdentificationService, mock_db: mock.Mock
) -> None:
    """Test identify with cache hit for known identity."""
    # Pre-populate cache with a known identity
    embedding = np.random.rand(512).astype(np.float32)
    embedding /= np.linalg.norm(embedding)

    face = Face(
        embedding=embedding,
        bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
        confidence=0.9,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        track_id=1,
    )

    # Add to cache
    cached_identity = CachedIdentity(
        name="John Doe", confidence=0.85, timestamp=time.time(), is_known=True
    )
    service._cache[1] = cached_identity

    # Call identify
    result = service.identify([face])

    # Assert mock_db.search_by_embedding was NOT called
    mock_db.search_by_embedding.assert_not_called()

    # Verify result
    assert len(result) == 1
    assert result[0].name == "John Doe"
    assert result[0].confidence == 0.85
    assert result[0].is_known is True
    assert result[0].track_id == 1


def test_identify_cache_hit_unknown_valid(
    service: IdentificationService, mock_db: mock.Mock
) -> None:
    """Test identify with cache hit for unknown identity (still valid)."""
    # Pre-populate cache with "Unknown" entry (timestamp = now)
    embedding = np.random.rand(512).astype(np.float32)
    embedding /= np.linalg.norm(embedding)

    face = Face(
        embedding=embedding,
        bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
        confidence=0.9,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        track_id=2,
    )

    # Add to cache as unknown
    cached_identity = CachedIdentity(
        name="Unknown", confidence=0.0, timestamp=time.time(), is_known=False
    )
    service._cache[2] = cached_identity

    # Call identify
    result = service.identify([face])

    # Assert mock_db.search_by_embedding was NOT called
    mock_db.search_by_embedding.assert_not_called()

    # Verify result
    assert len(result) == 1
    assert result[0].name == "Unknown"
    assert result[0].confidence == 0.0
    assert result[0].is_known is False
    assert result[0].track_id == 2


def test_identify_cache_miss_unknown_expired(
    service: IdentificationService, mock_db: mock.Mock
) -> None:
    """Test identify with cache miss for expired unknown identity."""
    # Pre-populate cache with "Unknown" entry (timestamp = now - TTL - 1s)
    embedding = np.random.rand(512).astype(np.float32)
    embedding /= np.linalg.norm(embedding)

    face = Face(
        embedding=embedding,
        bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
        confidence=0.9,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        track_id=3,
    )

    # Add to cache as unknown with old timestamp
    old_timestamp = time.time() - service.config.unknown_cache_ttl - 1  # Expired
    cached_identity = CachedIdentity(
        name="Unknown", confidence=0.0, timestamp=old_timestamp, is_known=False
    )
    service._cache[3] = cached_identity

    # Mock the database response
    mock_db.search_by_embedding.return_value = (None, 0.0)  # No match found

    # Call identify
    result = service.identify([face])

    # Assert mock_db.search_by_embedding WAS called
    mock_db.search_by_embedding.assert_called_once_with(
        embedding, service.config.similarity_threshold
    )

    # Verify result
    assert len(result) == 1
    assert result[0].name == "Unknown"
    assert result[0].confidence == 0.0
    assert result[0].is_known is False
    assert result[0].track_id == 3


def test_identify_db_query(service: IdentificationService, mock_db: mock.Mock) -> None:
    """Test the complete flow: Cache Miss -> DB Query -> Add to Cache -> Result."""
    # Ensure cache is empty for this track
    embedding = np.random.rand(512).astype(np.float32)
    embedding /= np.linalg.norm(embedding)

    face = Face(
        embedding=embedding,
        bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
        confidence=0.9,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        track_id=4,
    )

    # Mock the database response for a known person
    from src.face_recognize.database import PersonRecord

    mock_person = PersonRecord(
        id="test_id_123",
        name="Jane Smith",
        embedding=embedding,
        created_at="2026-01-30T10:15:23.456789",
    )
    mock_db.search_by_embedding.return_value = (mock_person, 0.88)

    # Call identify
    result = service.identify([face])

    # Verify database was queried
    mock_db.search_by_embedding.assert_called_once_with(
        embedding, service.config.similarity_threshold
    )

    # Verify result
    assert len(result) == 1
    assert result[0].name == "Jane Smith"
    assert result[0].confidence == 0.88
    assert result[0].is_known is True
    assert result[0].track_id == 4

    # Verify cache was updated
    assert 4 in service._cache
    cached_result = service._cache[4]
    assert cached_result.name == "Jane Smith"
    assert cached_result.confidence == 0.88
    assert cached_result.is_known is True
