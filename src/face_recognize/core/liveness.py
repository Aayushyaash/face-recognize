"""Liveness detection module for anti-spoofing using MiniFASNetV2 ONNX model.

This module implements a LivenessDetector that uses the MiniFASNetV2 model
to distinguish between real faces and spoof attempts (photos/screens).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

if TYPE_CHECKING:
    from face_recognize.core.models import BoundingBox


class LivenessDetector:
    """Liveness detection using MiniFASNetV2 ONNX model for anti-spoofing.

    The liveness detector distinguishes between real faces and spoof attempts
    (photos, screens) using a lightweight CNN model trained for this purpose.
    """

    def __init__(
        self,
        model_path: str | Path = "models/MiniFASNetV2.onnx",
        performance_mode: str = "balanced",  # "balanced", "performance", "accuracy"
    ) -> None:
        """Initialize the liveness detector with the ONNX model.

        Args:
            model_path: Path to the MiniFASNetV2 ONNX model file.
            performance_mode: Performance vs accuracy trade-off ("balanced",
                            "performance", "accuracy").
        """
        self.model_path = Path(model_path)

        # Configure ONNX Runtime session based on performance mode
        sess_options = ort.SessionOptions()

        if performance_mode == "performance":
            # Optimize for speed
            sess_options.intra_op_num_threads = os.cpu_count() or 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
        elif performance_mode == "accuracy":
            # Optimize for accuracy (may be slower)
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
        else:  # balanced
            # Balanced optimization
            sess_options.intra_op_num_threads = max(1, (os.cpu_count() or 1) // 2)
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            )

        # Create ONNX runtime session
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],  # Use CPU by default
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Model expects 80x80 RGB images
        self.input_shape = (1, 3, 80, 80)

    def predict(
        self, image: npt.NDArray[np.uint8], bbox: BoundingBox | None = None
    ) -> tuple[float, bool]:
        """Predict if a face is real or spoof.

        Args:
            image: Input image containing the face (BGR format).
            bbox: Optional bounding box to crop the face region.
                  If None, uses the entire image.

        Returns:
            Tuple of (liveness_score, is_real) where:
            - liveness_score: Float between 0.0 (spoof) and 1.0 (real)
            - is_real: Boolean indicating if face is considered real
        """
        # Preprocess the image
        preprocessed = self._preprocess(image, bbox)

        # Run inference
        result = self.session.run([self.output_name], {self.input_name: preprocessed})

        # Extract the prediction (assuming single output)
        prediction = result[0]  # Raw model output

        # Flatten to get scalar value
        prediction_flat = np.ravel(prediction)
        if len(prediction_flat) > 0:
            prediction_val = float(prediction_flat[0])
        else:
            # Default to 0.5 if no prediction returned
            prediction_val = 0.5

        # Convert to probability using sigmoid if needed
        # Some models output logits, others probabilities
        if prediction_val < 0 or prediction_val > 1:
            # Apply sigmoid to convert logits to probability
            liveness_score = 1.0 / (1.0 + np.exp(-prediction_val))
        else:
            # Already a probability
            liveness_score = prediction_val

        # Ensure score is in [0, 1] range
        liveness_score = max(0.0, min(1.0, liveness_score))

        # Determine if real based on score (0.5 threshold)
        is_real = bool(liveness_score > 0.5)

        return liveness_score, is_real

    def _preprocess(
        self, image: npt.NDArray[np.uint8], bbox: BoundingBox | None
    ) -> npt.NDArray[np.single]:
        """Preprocess the image for liveness detection.

        Args:
            image: Input image (BGR format).
            bbox: Optional bounding box to crop face region.

        Returns:
            Preprocessed image tensor ready for model inference.
        """
        # Crop the face region if bbox is provided
        if bbox is not None:
            face_region = image[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
        else:
            face_region = image

        # Resize to model input size (80x80)
        resized = cv2.resize(face_region, (80, 80))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize pixel values to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # Transpose to CHW format (channels first)
        chw = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batched = np.expand_dims(chw, axis=0)

        return batched.astype(np.float32)

    def change_model(self, model_path: str | Path) -> None:
        """Change the underlying ONNX model.

        Args:
            model_path: Path to the new ONNX model file.
        """
        self.model_path = Path(model_path)

        # Recreate the session with the new model
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = max(1, (os.cpu_count() or 1) // 2)
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # Update input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
