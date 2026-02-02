"""Test suite for face detection functionality."""

from unittest.mock import Mock, patch

import numpy as np

from src.face_recognize.config import DEFAULT_CONFIG
from src.face_recognize.core.detector import FaceDetector
from src.face_recognize.core.models import Face


class TestFaceDetector:
    """Test cases for FaceDetector class."""

    def test_initialization(self) -> None:
        """Test FaceDetector initialization with default config."""
        detector = FaceDetector(config=DEFAULT_CONFIG)

        assert detector.model_name == DEFAULT_CONFIG.model
        assert detector.threshold == DEFAULT_CONFIG.detection_threshold

    @patch("insightface.app.FaceAnalysis")
    def test_detect_faces_success(self, mock_face_analysis: Mock) -> None:
        """Test successful face detection."""
        # Mock the InsightFace app
        mock_app = Mock()
        mock_app.get.return_value = [
            {
                "bbox": np.array([10, 20, 110, 120]),  # [x1, y1, x2, y2]
                "kps": np.random.rand(5, 2),  # 5 facial landmarks
                "det_score": 0.85,  # Detection confidence
                "embedding": np.random.rand(512),  # 512-dim embedding
            }
        ]
        mock_face_analysis.return_value = mock_app

        detector = FaceDetector(config=DEFAULT_CONFIG)
        # Mock the prepare method to return the mock app
        detector.model = mock_app

        # Create a dummy image
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        faces = detector.detect_faces(dummy_img)

        assert len(faces) == 1
        face = faces[0]
        assert isinstance(face, Face)
        assert face.confidence == 0.85
        assert face.bbox.x1 == 10
        assert face.bbox.y1 == 20
        assert face.bbox.x2 == 110
        assert face.bbox.y2 == 120
        assert face.embedding.shape == (512,)

    @patch("insightface.app.FaceAnalysis")
    def test_detect_faces_below_threshold(self, mock_face_analysis: Mock) -> None:
        """Test face detection with confidence below threshold."""
        # Mock the InsightFace app to return a face with low confidence
        mock_app = Mock()
        mock_app.get.return_value = [
            {
                "bbox": np.array([10, 20, 110, 120]),
                "kps": np.random.rand(5, 2),
                "det_score": 0.2,  # Below default threshold of 0.7
                "embedding": np.random.rand(512),
            }
        ]
        mock_face_analysis.return_value = mock_app

        detector = FaceDetector(config=DEFAULT_CONFIG)
        detector.model = mock_app

        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        faces = detector.detect_faces(dummy_img)

        # With threshold at 0.7, the face with 0.2 confidence should be filtered out
        assert len(faces) == 0

    @patch("insightface.app.FaceAnalysis")
    def test_detect_faces_multiple_faces(self, mock_face_analysis: Mock) -> None:
        """Test face detection with multiple faces."""
        # Mock the InsightFace app to return multiple faces
        mock_app = Mock()
        mock_app.get.return_value = [
            {
                "bbox": np.array([10, 20, 110, 120]),
                "kps": np.random.rand(5, 2),
                "det_score": 0.85,
                "embedding": np.random.rand(512),
            },
            {
                "bbox": np.array([200, 150, 300, 250]),
                "kps": np.random.rand(5, 2),
                "det_score": 0.92,
                "embedding": np.random.rand(512),
            },
        ]
        mock_face_analysis.return_value = mock_app

        detector = FaceDetector(config=DEFAULT_CONFIG)
        detector.model = mock_app

        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        faces = detector.detect_faces(dummy_img)

        assert len(faces) == 2
        # Both faces should have confidence above threshold
        for face in faces:
            assert face.confidence >= 0.7

    @patch("insightface.app.FaceAnalysis")
    def test_detect_faces_none_found(self, mock_face_analysis: Mock) -> None:
        """Test face detection when no faces are found."""
        # Mock the InsightFace app to return no faces
        mock_app = Mock()
        mock_app.get.return_value = []
        mock_face_analysis.return_value = mock_app

        detector = FaceDetector(config=DEFAULT_CONFIG)
        detector.model = mock_app

        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        faces = detector.detect_faces(dummy_img)

        assert len(faces) == 0
