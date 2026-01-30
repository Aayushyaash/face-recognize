"""Unit tests for face detection functionality."""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from src.face_recognize.config import AppConfig, DEFAULT_CONFIG
from src.face_recognize.core.detector import FaceDetector
from src.face_recognize.core.models import BoundingBox, Face


class TestFaceDetectionFunctionality:
    """Test cases for face detection functionality."""

    @patch('insightface.app.FaceAnalysis')
    def test_detect_faces_method(self, mock_face_analysis):
        """Test the detect_faces method."""
        # Mock the InsightFace app
        mock_app = Mock()
        mock_app.get.return_value = [
            {
                'bbox': np.array([10, 20, 110, 120]),  # [x1, y1, x2, y2]
                'kps': np.random.rand(5, 2),  # 5 facial landmarks
                'det_score': 0.85,  # Detection confidence
                'embedding': np.random.rand(512)  # 512-dim embedding
            }
        ]
        mock_face_analysis.return_value = mock_app

        detector = FaceDetector(config=DEFAULT_CONFIG)
        # Manually set the model to the mock for testing
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

    @patch('insightface.app.FaceAnalysis')
    def test_detect_faces_with_boxes_and_confidence_method(self, mock_face_analysis):
        """Test the detect_faces_with_boxes_and_confidence method."""
        # Mock the InsightFace app
        mock_app = Mock()
        mock_app.get.return_value = [
            {
                'bbox': np.array([10, 20, 110, 120]),
                'kps': np.random.rand(5, 2),
                'det_score': 0.85,
                'embedding': np.random.rand(512)
            },
            {
                'bbox': np.array([200, 150, 300, 250]),
                'kps': np.random.rand(5, 2),
                'det_score': 0.92,
                'embedding': np.random.rand(512)
            }
        ]
        mock_face_analysis.return_value = mock_app

        detector = FaceDetector(config=DEFAULT_CONFIG)
        detector.model = mock_app

        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        bboxes, confidences = detector.detect_faces_with_boxes_and_confidence(dummy_img)

        assert len(bboxes) == 2
        assert len(confidences) == 2
        assert isinstance(bboxes[0], BoundingBox)
        assert isinstance(confidences[0], float)
        assert confidences[0] >= 0.7  # Above threshold
        assert confidences[1] >= 0.7  # Above threshold

    def test_set_and_get_threshold(self):
        """Test setting and getting the detection threshold."""
        config = AppConfig(detection_threshold=0.7)
        detector = FaceDetector(config=config)

        # Initial threshold should be 0.7
        assert detector.get_threshold() == 0.7

        # Change threshold to 0.5
        detector.set_threshold(0.5)
        assert detector.get_threshold() == 0.5

        # Change threshold to 0.9
        detector.set_threshold(0.9)
        assert detector.get_threshold() == 0.9

    def test_set_threshold_invalid_values(self):
        """Test setting invalid threshold values."""
        config = AppConfig(detection_threshold=0.7)
        detector = FaceDetector(config=config)

        # Test value below 0
        with pytest.raises(ValueError):
            detector.set_threshold(-0.1)

        # Test value above 1
        with pytest.raises(ValueError):
            detector.set_threshold(1.1)

        # Threshold should still be the original value
        assert detector.get_threshold() == 0.7

    @patch('insightface.app.FaceAnalysis')
    def test_change_model(self, mock_face_analysis):
        """Test changing the InsightFace model."""
        # Mock the InsightFace app
        mock_app = Mock()
        mock_face_analysis.return_value = mock_app

        detector = FaceDetector(config=DEFAULT_CONFIG)

        # Initially should be using the default model
        assert detector.model_name == DEFAULT_CONFIG.model

        # Change to buffalo_l
        detector.change_model('buffalo_l')
        assert detector.model_name == 'buffalo_l'

        # Change to buffalo_sc
        detector.change_model('buffalo_sc')
        assert detector.model_name == 'buffalo_sc'

    def test_change_model_invalid_name(self):
        """Test changing to an unsupported model."""
        config = AppConfig(model='buffalo_s')
        detector = FaceDetector(config=config)

        # Try to change to an unsupported model
        with pytest.raises(ValueError):
            detector.change_model('unsupported_model')

        # Model should still be the original
        assert detector.model_name == 'buffalo_s'