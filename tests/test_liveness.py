"""Unit tests for LivenessDetector."""

from unittest.mock import Mock, patch

import numpy as np

from face_recognize.core.liveness import LivenessDetector
from face_recognize.core.models import BoundingBox


class TestLivenessDetector:
    """Test cases for LivenessDetector class."""

    @patch("onnxruntime.InferenceSession")
    def test_initialization(self, mock_session: Mock) -> None:
        """Test initialization with default parameters."""
        # Mock the session to avoid needing an actual ONNX model
        mock_session_instance = Mock()
        mock_session_instance.get_inputs.return_value = [Mock()]
        mock_session_instance.get_inputs.return_value[0].name = "input"
        mock_session_instance.get_outputs.return_value = [Mock()]
        mock_session_instance.get_outputs.return_value[0].name = "output"
        mock_session.return_value = mock_session_instance

        detector = LivenessDetector()

        assert detector.model_path.name == "MiniFASNetV2.onnx"
        mock_session.assert_called_once()

    @patch("onnxruntime.InferenceSession")
    def test_initialization_with_custom_params(self, mock_session: Mock) -> None:
        """Test initialization with custom model path and performance mode."""
        mock_session_instance = Mock()
        mock_session_instance.get_inputs.return_value = [Mock()]
        mock_session_instance.get_inputs.return_value[0].name = "input"
        mock_session_instance.get_outputs.return_value = [Mock()]
        mock_session_instance.get_outputs.return_value[0].name = "output"
        mock_session.return_value = mock_session_instance

        detector = LivenessDetector(
            model_path="custom/model.onnx", performance_mode="performance"
        )

        assert detector.model_path.name == "model.onnx"
        mock_session.assert_called_once()

    @patch("onnxruntime.InferenceSession")
    def test_predict_with_bbox(self, mock_session: Mock) -> None:
        """Test prediction with bounding box."""
        # Setup mock
        mock_session_instance = Mock()
        mock_session_instance.get_inputs.return_value = [Mock()]
        mock_session_instance.get_inputs.return_value[0].name = "input"
        mock_session_instance.get_outputs.return_value = [Mock()]
        mock_session_instance.get_outputs.return_value[0].name = "output"
        mock_session_instance.run.return_value = [[np.array([[0.8]])]]
        mock_session.return_value = mock_session_instance

        detector = LivenessDetector()

        # Create test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(x1=20, y1=20, x2=80, y2=80)

        liveness_score, is_real = detector.predict(image, bbox)

        assert 0.0 <= liveness_score <= 1.0
        assert isinstance(is_real, bool)
        # Since the mock returns 0.8, is_real should be True (above 0.5 threshold)
        assert is_real is True

    @patch("onnxruntime.InferenceSession")
    def test_predict_without_bbox(self, mock_session: Mock) -> None:
        """Test prediction on entire image."""
        # Setup mock
        mock_session_instance = Mock()
        mock_session_instance.get_inputs.return_value = [Mock()]
        mock_session_instance.get_inputs.return_value[0].name = "input"
        mock_session_instance.get_outputs.return_value = [Mock()]
        mock_session_instance.get_outputs.return_value[0].name = "output"
        mock_session_instance.run.return_value = [[np.array([[0.3]])]]
        mock_session.return_value = mock_session_instance

        detector = LivenessDetector()

        # Create test image
        image = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)

        liveness_score, is_real = detector.predict(image)

        assert 0.0 <= liveness_score <= 1.0
        assert isinstance(is_real, bool)
        # Since the mock returns 0.3, is_real should be False (below 0.5 threshold)
        assert is_real is False

    @patch("onnxruntime.InferenceSession")
    def test_preprocess_method(self, mock_session: Mock) -> None:
        """Test the preprocessing method."""
        # Setup mock
        mock_session_instance = Mock()
        mock_session_instance.get_inputs.return_value = [Mock()]
        mock_session_instance.get_inputs.return_value[0].name = "input"
        mock_session_instance.get_outputs.return_value = [Mock()]
        mock_session_instance.get_outputs.return_value[0].name = "output"
        mock_session.return_value = mock_session_instance

        detector = LivenessDetector()

        # Create test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        bbox = BoundingBox(x1=20, y1=20, x2=80, y2=80)

        preprocessed = detector._preprocess(image, bbox)

        # Check that the output has the expected shape (batch, channels, height, width)
        assert preprocessed.shape == (1, 3, 80, 80)
        assert preprocessed.dtype == np.float32

    @patch("onnxruntime.InferenceSession")
    def test_prediction_with_logits_output(self, mock_session: Mock) -> None:
        """Test prediction when model outputs logits."""
        # Setup mock to return a logit value
        mock_session_instance = Mock()
        mock_session_instance.get_inputs.return_value = [Mock()]
        mock_session_instance.get_inputs.return_value[0].name = "input"
        mock_session_instance.get_outputs.return_value = [Mock()]
        mock_session_instance.get_outputs.return_value[0].name = "output"
        mock_session_instance.run.return_value = [[np.array([[-1.0]])]]  # Logit value
        mock_session.return_value = mock_session_instance

        detector = LivenessDetector()

        # Create test image
        image = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)

        liveness_score, is_real = detector.predict(image)

        # With logit of -1.0, sigmoid(-1.0) ≈ 0.269, so is_real should be False
        assert 0.0 <= liveness_score <= 1.0
        assert is_real is False

    @patch("onnxruntime.InferenceSession")
    def test_change_model(self, mock_session: Mock) -> None:
        """Test changing the underlying model."""
        # Setup mock
        mock_session_instance = Mock()
        mock_session_instance.get_inputs.return_value = [Mock()]
        mock_session_instance.get_inputs.return_value[0].name = "input"
        mock_session_instance.get_outputs.return_value = [Mock()]
        mock_session_instance.get_outputs.return_value[0].name = "output"
        mock_session.return_value = mock_session_instance

        detector = LivenessDetector()

        # Change model
        detector.change_model("new_model.onnx")

        # Verify the session was recreated
        assert mock_session.call_count == 2  # Initial + change_model
