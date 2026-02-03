"""Unit tests for FaceTracker class."""

from pathlib import Path

import numpy as np
import pytest
from face_recognize.config import AppConfig
from face_recognize.core.models import BoundingBox, Face
from face_recognize.core.tracker import FaceTracker


@pytest.fixture
def sample_config() -> AppConfig:
    """Returns AppConfig with known thresholds."""
    return AppConfig(
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


@pytest.fixture
def mock_faces() -> list[Face]:
    """Returns a list of Face objects for testing updates."""
    embedding1 = np.random.rand(512).astype(np.float32)
    embedding1 /= np.linalg.norm(embedding1)  # Normalize

    embedding2 = np.random.rand(512).astype(np.float32)
    embedding2 /= np.linalg.norm(embedding2)  # Normalize

    embedding3 = np.random.rand(512).astype(np.float32)
    embedding3 /= np.linalg.norm(embedding3)  # Normalize

    return [
        Face(
            embedding=embedding1,
            bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
            confidence=0.9,
            landmarks=np.zeros((5, 2), dtype=np.float32),
            track_id=None,
        ),
        Face(
            embedding=embedding2,
            bbox=BoundingBox(x1=100, y1=100, x2=150, y2=150),
            confidence=0.85,
            landmarks=np.zeros((5, 2), dtype=np.float32),
            track_id=None,
        ),
        Face(
            embedding=embedding3,
            bbox=BoundingBox(x1=200, y1=200, x2=250, y2=250),
            confidence=0.8,
            landmarks=np.zeros((5, 2), dtype=np.float32),
            track_id=None,
        ),
    ]


def test_iou_calculation() -> None:
    """Test BoundingBox.iou calculation."""
    # Identical boxes should have IoU of 1.0
    box1 = BoundingBox(x1=10, y1=10, x2=50, y2=50)
    box2 = BoundingBox(x1=10, y1=10, x2=50, y2=50)
    assert box1.iou(box2) == 1.0
    assert box2.iou(box1) == 1.0

    # Disjoint boxes should have IoU of 0.0
    box3 = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    box4 = BoundingBox(x1=20, y1=20, x2=30, y2=30)
    assert box3.iou(box4) == 0.0
    assert box4.iou(box3) == 0.0

    # Partially overlapping boxes
    box5 = BoundingBox(x1=0, y1=0, x2=20, y2=20)  # Area = 400
    box6 = BoundingBox(x1=10, y1=10, x2=30, y2=30)  # Area = 400
    # Intersection: (10,10) to (20,20) = 100, Union: 400+400-100 = 700
    expected_iou = 100 / 700
    assert abs(box5.iou(box6) - expected_iou) < 1e-9
    assert abs(box6.iou(box5) - expected_iou) < 1e-9


def test_tracker_update_initial(
    sample_config: AppConfig, mock_faces: list[Face]
) -> None:
    """Test tracker.update with initial faces."""
    tracker = FaceTracker(sample_config)

    result = tracker.update(mock_faces)

    # Should return 3 faces with track_id 1, 2, 3
    assert len(result) == 3
    assert result[0].track_id == 1
    assert result[1].track_id == 2
    assert result[2].track_id == 3

    # Internal state should have 3 tracks
    assert tracker.get_active_track_count() == 3


def test_tracker_match_logic(sample_config: AppConfig) -> None:
    """Test tracker match logic with shifted faces."""
    tracker = FaceTracker(sample_config)

    # Create initial faces
    embedding1 = np.random.rand(512).astype(np.float32)
    embedding1 /= np.linalg.norm(embedding1)

    face_a_original = Face(
        embedding=embedding1,
        bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
        confidence=0.9,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        track_id=None,
    )

    # Stage 1: Setup tracks for face A
    result1 = tracker.update([face_a_original])
    original_track_id = result1[0].track_id

    # Stage 2: Call update with Face A shifted slightly (same embedding)
    face_a_shifted = Face(
        embedding=embedding1,  # Same embedding
        bbox=BoundingBox(x1=12, y1=12, x2=52, y2=52),  # Slightly shifted
        confidence=0.9,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        track_id=None,
    )

    result2 = tracker.update([face_a_shifted])

    # Track ID should remain consistent
    assert result2[0].track_id == original_track_id

    # track.hits should increment
    # We can't directly access the track, but we can verify the behavior by
    # checking that the same ID is preserved across frames


def test_tracker_embedding_confirmation(sample_config: AppConfig) -> None:
    """Test tracker creates new track for high IoU but different embedding."""
    from dataclasses import replace

    # Create a config with a higher similarity threshold to ensure
    # embedding verification matters
    strict_config = replace(sample_config, similarity_threshold=0.9)
    # Higher threshold to make verification more stringent

    tracker = FaceTracker(strict_config)

    # Create original face
    embedding1 = np.random.rand(512).astype(np.float32)
    embedding1 /= np.linalg.norm(embedding1)

    face_a = Face(
        embedding=embedding1,
        bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
        confidence=0.9,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        track_id=None,
    )

    # Setup track for face A
    result1 = tracker.update([face_a])
    original_track_id = result1[0].track_id

    # Create face A' with high IoU but significantly different embedding
    # Generate a completely different random embedding
    embedding2 = np.random.rand(512).astype(np.float32)
    embedding2 /= np.linalg.norm(embedding2)

    face_a_prime = Face(
        embedding=embedding2,  # Different embedding
        bbox=BoundingBox(x1=11, y1=11, x2=51, y2=51),  # High IoU with original
        confidence=0.9,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        track_id=None,
    )

    result2 = tracker.update([face_a_prime])

    # Should create a new track ID (imposter/mismatch detected)
    # The IoU will be high enough to attempt matching, but
    # similarity will be below threshold
    assert result2[0].track_id != original_track_id


def test_tracker_aging(sample_config: AppConfig) -> None:
    """Test tracker aging and removal of stale tracks."""
    tracker = FaceTracker(sample_config)

    # Add a track
    embedding = np.random.rand(512).astype(np.float32)
    embedding /= np.linalg.norm(embedding)

    face = Face(
        embedding=embedding,
        bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50),
        confidence=0.9,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        track_id=None,
    )

    result = tracker.update([face])
    original_track_id = result[0].track_id

    # Verify track exists and has the expected ID
    assert tracker.get_active_track_count() == 1
    assert original_track_id is not None

    # Call update([]) max_track_age times - track should remain
    for _ in range(sample_config.max_track_age - 1):
        tracker.update([])
        assert tracker.get_active_track_count() == 1

    # Call update([]) one more time - track should be removed
    tracker.update([])
    assert tracker.get_active_track_count() == 0
