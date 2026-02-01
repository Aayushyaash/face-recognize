"""Integration tests with sample images for face detection module."""

from unittest.mock import patch

import numpy as np

from src.face_recognize.config import DEFAULT_CONFIG
from src.face_recognize.core.detector import FaceDetector


def test_integration_with_sample_images():
    """Perform integration testing with sample images."""
    detector = FaceDetector(config=DEFAULT_CONFIG)

    # Create a synthetic test image (normally we'd load a real image)
    # For integration testing without downloading models, we'll simulate
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Mock the model.get method to simulate face detection
    with patch.object(detector.model, "get") as mock_get:
        # Simulate detection of 2 faces
        mock_face1 = type("MockFace", (), {})()
        mock_face1.det_score = 0.85
        mock_face1.bbox = np.array([100, 100, 200, 200])
        mock_face1.kps = np.random.rand(5, 2)
        mock_face1.embedding = np.random.rand(512)

        mock_face2 = type("MockFace", (), {})()
        mock_face2.det_score = 0.92
        mock_face2.bbox = np.array([300, 150, 400, 250])
        mock_face2.kps = np.random.rand(5, 2)
        mock_face2.embedding = np.random.rand(512)

        mock_get.return_value = [mock_face1, mock_face2]

        # Run detection
        faces = detector.detect_faces(test_img)

        # Verify results
        assert len(faces) == 2
        for face in faces:
            assert face.confidence >= 0.7  # Meets threshold
            assert face.bbox.x1 >= 0
            assert face.bbox.y1 >= 0
            assert face.embedding.shape == (512,)


def test_integration_different_thresholds():
    """Test integration with different confidence thresholds."""
    # Start with default config
    detector = FaceDetector(config=DEFAULT_CONFIG)

    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    with patch.object(detector.model, "get") as mock_get:
        # Simulate detection of faces with different confidence scores
        mock_face_high = type("MockFace", (), {})()
        mock_face_high.det_score = 0.90
        mock_face_high.bbox = np.array([50, 50, 150, 150])
        mock_face_high.kps = np.random.rand(5, 2)
        mock_face_high.embedding = np.random.rand(512)

        mock_face_low = type("MockFace", (), {})()
        mock_face_low.det_score = 0.60  # Below default threshold of 0.7
        mock_face_low.bbox = np.array([200, 200, 300, 300])
        mock_face_low.kps = np.random.rand(5, 2)
        mock_face_low.embedding = np.random.rand(512)

        mock_get.return_value = [mock_face_high, mock_face_low]

        # With default threshold (0.7), should only get the high-confidence face
        faces = detector.detect_faces(test_img)
        assert len(faces) == 1
        assert faces[0].confidence == 0.90

        # Change threshold to 0.5, should now get both faces
        detector.set_threshold(0.5)
        faces = detector.detect_faces(test_img)
        assert len(faces) == 2


def test_integration_multiple_models():
    """Test integration with different models."""
    detector = FaceDetector(config=DEFAULT_CONFIG)

    # Test changing models
    original_model = detector.model_name
    assert original_model == DEFAULT_CONFIG.model

    # Change to buffalo_l
    detector.change_model("buffalo_l")
    assert detector.model_name == "buffalo_l"

    # Change to buffalo_sc
    detector.change_model("buffalo_sc")
    assert detector.model_name == "buffalo_sc"

    # Verify it still works after model change
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    with patch.object(detector.model, "get") as mock_get:
        mock_face = type("MockFace", (), {})()
        mock_face.det_score = 0.80
        mock_face.bbox = np.array([100, 100, 200, 200])
        mock_face.kps = np.random.rand(5, 2)
        mock_face.embedding = np.random.rand(512)

        mock_get.return_value = [mock_face]

        faces = detector.detect_faces(test_img)
        assert len(faces) == 1
        assert faces[0].confidence == 0.80


if __name__ == "__main__":
    test_integration_with_sample_images()
    test_integration_different_thresholds()
    test_integration_multiple_models()
    print("All integration tests passed!")
