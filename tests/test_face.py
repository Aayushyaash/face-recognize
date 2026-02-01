"""Unit tests for Face class."""

import numpy as np

from src.face_recognize.core.models import BoundingBox, Face


class TestFace:
    """Test cases for Face class."""

    def test_similarity_with_another_face(self):
        """Test similarity calculation with another face."""
        # Create two faces with normalized embeddings
        embedding1 = np.array([1.0, 0.0, 0.0])  # Normalized
        embedding2 = np.array(
            [0.8, 0.6, 0.0]
        )  # Also normalized (sqrt(0.8^2 + 0.6^2) = 1.0)

        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        landmarks = np.zeros((5, 2))  # 5 landmarks with 2 coordinates each

        face1 = Face(
            embedding=embedding1, bbox=bbox, confidence=0.9, landmarks=landmarks
        )
        face2 = Face(
            embedding=embedding2, bbox=bbox, confidence=0.8, landmarks=landmarks
        )

        # Cosine similarity: [1,0,0] · [0.8,0.6,0] = 0.8
        expected_similarity = 0.8
        calculated_similarity = face1.similarity(face2)

        assert abs(calculated_similarity - expected_similarity) < 0.001

    def test_similarity_to_embedding(self):
        """Test similarity calculation with an embedding vector."""
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([0.6, 0.8, 0.0])  # Normalized

        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        landmarks = np.zeros((5, 2))

        face = Face(
            embedding=embedding1, bbox=bbox, confidence=0.9, landmarks=landmarks
        )

        # Cosine similarity: [1,0,0] · [0.6,0.8,0] = 0.6
        expected_similarity = 0.6
        calculated_similarity = face.similarity_to_embedding(embedding2)

        assert abs(calculated_similarity - expected_similarity) < 0.001
