"""Unit tests for error handling in face detection."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.face_recognize.config import DEFAULT_CONFIG
from src.face_recognize.core.detector import FaceDetector


class TestFaceDetectionErrorHandling:
    """Unit tests for error handling in face detection."""

    def test_detect_faces_empty_image(self):
        """Test detecting faces in an empty image."""
        detector = FaceDetector(config=DEFAULT_CONFIG)

        # Empty image
        empty_img = np.array([])
        faces = detector.detect_faces(empty_img)
        assert len(faces) == 0

    def test_detect_faces_invalid_dimensions(self):
        """Test detecting faces in an image with invalid dimensions."""
        detector = FaceDetector(config=DEFAULT_CONFIG)

        # 1D array (invalid image)
        invalid_img = np.array([1, 2, 3])
        faces = detector.detect_faces(invalid_img)
        assert len(faces) == 0

    @patch("insightface.app.FaceAnalysis")
    def test_detect_faces_model_failure(self, mock_face_analysis):
        """Test detecting faces when the model fails."""
        # Mock the InsightFace app to raise an exception
        mock_app = Mock()
        mock_app.get.side_effect = Exception("Model failed to process image")
        mock_face_analysis.return_value = mock_app

        detector = FaceDetector(config=DEFAULT_CONFIG)
        detector.model = mock_app

        # Create a dummy image
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Should return empty list instead of raising an exception
        faces = detector.detect_faces(dummy_img)
        assert len(faces) == 0

    @patch("insightface.app.FaceAnalysis")
    def test_detect_faces_with_invalid_bbox(self, mock_face_analysis):
        """Test detecting faces when bounding box extraction fails."""
        # Mock the InsightFace app to return a face with invalid bbox
        mock_app = Mock()
        mock_face_info = Mock()
        mock_face_info.det_score = 0.85
        mock_face_info.bbox = "invalid_bbox"  # Invalid bbox that will cause an error
        mock_face_info.kps = np.random.rand(5, 2)
        mock_face_info.embedding = np.random.rand(512)
        mock_app.get.return_value = [mock_face_info]
        mock_face_analysis.return_value = mock_app

        detector = FaceDetector(config=DEFAULT_CONFIG)
        detector.model = mock_app

        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Should return empty list since bbox extraction fails
        faces = detector.detect_faces(dummy_img)
        assert len(faces) == 0

    @patch("insightface.app.FaceAnalysis")
    def test_detect_faces_with_invalid_landmarks(self, mock_face_analysis):
        """Test detecting faces when landmark extraction fails."""
        # Mock the InsightFace app to return a face with invalid landmarks
        mock_app = Mock()
        mock_face_info = Mock()
        mock_face_info.det_score = 0.85
        mock_face_info.bbox = np.array([10, 20, 110, 120])
        mock_face_info.kps = "invalid_kps"  # Invalid kps that will cause an error
        mock_face_info.embedding = np.random.rand(512)
        mock_app.get.return_value = [mock_face_info]
        mock_face_analysis.return_value = mock_app

        detector = FaceDetector(config=DEFAULT_CONFIG)
        detector.model = mock_app

        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Should return empty list since landmark extraction fails
        faces = detector.detect_faces(dummy_img)
        assert len(faces) == 0

    @patch("insightface.app.FaceAnalysis")
    def test_detect_faces_with_invalid_embedding(self, mock_face_analysis):
        """Test detecting faces when embedding extraction fails."""
        # Mock the InsightFace app to return a face with invalid embedding
        mock_app = Mock()
        mock_face_info = Mock()
        mock_face_info.det_score = 0.85
        mock_face_info.bbox = np.array([10, 20, 110, 120])
        mock_face_info.kps = np.random.rand(5, 2)
        mock_face_info.embedding = (
            "invalid_embedding"  # Invalid embedding that will cause an error
        )
        mock_app.get.return_value = [mock_face_info]
        mock_face_analysis.return_value = mock_app

        detector = FaceDetector(config=DEFAULT_CONFIG)
        detector.model = mock_app

        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Should return empty list since embedding extraction fails
        faces = detector.detect_faces(dummy_img)
        assert len(faces) == 0

    @patch("insightface.app.FaceAnalysis")
    def test_detect_faces_with_zero_norm_embedding(self, mock_face_analysis):
        """Test faces when embedding has zero norm (ZeroDivisionError)."""
        # Mock the InsightFace app to return a face with zero embedding
        mock_app = Mock()
        mock_face_info = Mock()
        mock_face_info.det_score = 0.85
        mock_face_info.bbox = np.array([10, 20, 110, 120])
        mock_face_info.kps = np.random.rand(5, 2)
        mock_face_info.embedding = np.zeros(512)  # Zero norm embedding
        mock_app.get.return_value = [mock_face_info]
        mock_face_analysis.return_value = mock_app

        detector = FaceDetector(config=DEFAULT_CONFIG)
        detector.model = mock_app

        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Should return empty list since embedding normalization fails
        faces = detector.detect_faces(dummy_img)
        assert len(faces) == 0

    def test_grayscale_image_conversion(self):
        """Test that grayscale images are properly converted."""
        detector = FaceDetector(config=DEFAULT_CONFIG)

        # Grayscale image
        gray_img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

        # Should not raise an error when converting grayscale to RGB
        try:
            with patch.object(detector.model, "get", return_value=[]):
                faces = detector.detect_faces(gray_img)
                assert isinstance(faces, list)
        except Exception:
            pytest.fail("Grayscale image conversion should not raise an error")
