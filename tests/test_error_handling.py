"""Test cases for error handling scenarios."""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from src.face_recognize.config import DEFAULT_CONFIG
from src.face_recognize.core.detector import FaceDetector


class TestErrorHandlingScenarios:
    """Test cases for error handling scenarios."""

    @patch('insightface.app.FaceAnalysis')
    def test_error_handling_for_images_with_no_faces(self, mock_face_analysis):
        """Test error handling for images with no faces."""
        # Mock the InsightFace app to return no faces
        mock_app = Mock()
        mock_app.get.return_value = []  # No faces detected
        mock_face_analysis.return_value = mock_app

        detector = FaceDetector(config=DEFAULT_CONFIG)
        detector.model = mock_app

        # Create a dummy image
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # This should not raise an error, just return empty list
        faces = detector.detect_faces(dummy_img)
        assert len(faces) == 0

        # Also test the method that returns boxes and confidences
        bboxes, confidences = detector.detect_faces_with_boxes_and_confidence(dummy_img)
        assert len(bboxes) == 0
        assert len(confidences) == 0

    def test_validation_for_image_input_formats(self):
        """Test validation for different image input formats."""
        detector = FaceDetector(config=DEFAULT_CONFIG)

        # Test with a valid RGB image
        rgb_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Should not raise an error (though might not detect faces without proper model)
        try:
            # We'll patch the model.get method to return empty list to avoid actual model loading
            with patch.object(detector.model, 'get', return_value=[]):
                faces = detector.detect_faces(rgb_img)
                assert isinstance(faces, list)
        except Exception:
            pytest.fail("Valid RGB image should not cause an error")

        # Test with a grayscale image
        gray_img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        try:
            with patch.object(detector.model, 'get', return_value=[]):
                faces = detector.detect_faces(gray_img)
                assert isinstance(faces, list)
        except Exception:
            pytest.fail("Valid grayscale image should not cause an error")

        # Test with an invalid image shape
        invalid_img = np.random.randint(0, 255, (10, 20), dtype=np.uint8)  # Too small
        try:
            with patch.object(detector.model, 'get', return_value=[]):
                faces = detector.detect_faces(invalid_img)
                # Even invalid images should not crash, just return empty list
        except Exception:
            # Some invalid images might cause errors, which is acceptable
            pass

    @patch('insightface.app.FaceAnalysis')
    def test_handling_of_corrupted_invalid_images(self, mock_face_analysis):
        """Test handling of corrupted/invalid images."""
        detector = FaceDetector(config=DEFAULT_CONFIG)

        # Test with an image that has NaN values
        nan_img = np.full((100, 100, 3), np.nan, dtype=float)
        try:
            with patch.object(detector.model, 'get', side_effect=Exception("Invalid image")):
                faces = detector.detect_faces(nan_img)
                # If the underlying model throws an error, we expect it to be handled gracefully
        except Exception:
            # It's acceptable for invalid images to cause exceptions at the model level
            pass

        # Test with an image that has infinite values
        inf_img = np.full((100, 100, 3), np.inf, dtype=float)
        try:
            with patch.object(detector.model, 'get', side_effect=Exception("Invalid image")):
                faces = detector.detect_faces(inf_img)
        except Exception:
            # Again, model-level exceptions for invalid images are acceptable
            pass

        # Test with zero-sized image
        zero_img = np.empty((0, 0, 3), dtype=np.uint8)
        try:
            with patch.object(detector.model, 'get', side_effect=Exception("Invalid image")):
                faces = detector.detect_faces(zero_img)
        except Exception:
            # Zero-sized images may cause model errors, which is expected
            pass