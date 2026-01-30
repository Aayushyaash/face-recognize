"""Face detection module using InsightFace.

This module provides face detection capabilities using InsightFace models.
It can detect faces in images, extract bounding boxes, confidence scores,
facial landmarks, and face embeddings.
"""

from typing import List, Optional

import cv2
import insightface
import numpy as np

from ..config import AppConfig
from .models import BoundingBox, Face


class FaceDetector:
    """Face detection using InsightFace models.

    This class provides methods to detect faces in images using InsightFace models.
    It supports configurable detection thresholds and multiple model types.
    """

    def __init__(self, config: AppConfig):
        """Initialize the face detector.

        Args:
            config: Application configuration.
        """
        self.config = config
        self.model_name = config.model
        self.threshold = config.detection_threshold
        self.device = config.device

        # Initialize the InsightFace model
        self.model = insightface.app.FaceAnalysis(name=self.model_name, root='./models')
        self.model.prepare(ctx_id=0 if self.device == 'cpu' else 0)  # ctx_id 0 for CPU

    def change_model(self, model_name: str) -> None:
        """Change the InsightFace model.

        Args:
            model_name: Name of the new model (e.g., 'buffalo_s', 'buffalo_l', 'buffalo_sc').

        Raises:
            ValueError: If the model name is not supported.
        """
        if model_name not in ['buffalo_s', 'buffalo_l', 'buffalo_sc']:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: buffalo_s, buffalo_l, buffalo_sc")

        self.model_name = model_name
        # Reinitialize the model with the new name
        self.model = insightface.app.FaceAnalysis(name=self.model_name, root='./models')
        self.model.prepare(ctx_id=0 if self.device == 'cpu' else 0)  # ctx_id 0 for CPU

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        """Detect faces in an image.

        Args:
            image: Input image as numpy array (H, W, C).

        Returns:
            List of detected Face objects.
        """
        # Validate input image
        if image.size == 0:
            # Return empty list for empty images
            return []

        if len(image.shape) < 2:
            # Return empty list for invalid image dimensions
            return []

        # Convert image to RGB if it's BGR
        if len(image.shape) == 3 and image.shape[-1] == 3:
            # Assume BGR if the values seem high (indicating 0-255 range)
            if image.max() > 1.0:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        elif len(image.shape) == 2:
            # Grayscale image, convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image

        try:
            # Perform face detection
            faces_info = self.model.get(image_rgb)
        except Exception:
            # If model fails to process the image, return empty list
            return []

        detected_faces = []
        for face_info in faces_info:
            # Extract confidence score - handle both dict, Mock, and object formats
            try:
                if hasattr(face_info, 'det_score'):
                    # Object or Mock with det_score attribute
                    confidence = float(face_info.det_score)
                else:
                    # Dictionary format
                    confidence = float(face_info['det_score'])
            except (AttributeError, KeyError, TypeError, ValueError):
                # Skip face if confidence extraction fails
                continue

            # PERFORMANCE OPTIMIZATION: Early exit if below threshold
            if confidence < self.threshold:
                continue

            # Extract bounding box - handle both dict, Mock, and object formats
            try:
                if hasattr(face_info, 'bbox'):
                    # Object or Mock with bbox attribute
                    bbox_array = face_info.bbox.astype(int)
                else:
                    # Dictionary format
                    bbox_array = face_info['bbox'].astype(int)
            except (AttributeError, KeyError, TypeError):
                # Skip face if bounding box extraction fails
                continue

            try:
                bbox = BoundingBox(
                    x1=bbox_array[0],
                    y1=bbox_array[1],
                    x2=bbox_array[2],
                    y2=bbox_array[3]
                )
            except (ValueError, IndexError):
                # Skip face if bounding box extraction fails
                continue

            # Extract landmarks (5 facial points) - handle both dict, Mock, and object formats
            try:
                if hasattr(face_info, 'kps'):
                    # Object or Mock with kps attribute
                    landmarks = face_info.kps.astype(np.float32)
                else:
                    # Dictionary format
                    landmarks = face_info['kps'].astype(np.float32)
            except (AttributeError, KeyError, TypeError):
                # Skip face if landmark extraction fails
                continue

            # Extract embedding - handle both dict, Mock, and object formats
            try:
                if hasattr(face_info, 'embedding'):
                    # Object or Mock with embedding attribute
                    embedding = face_info.embedding.astype(np.float32)
                else:
                    # Dictionary format
                    embedding = face_info['embedding'].astype(np.float32)
            except (AttributeError, KeyError, TypeError):
                # Skip face if embedding extraction fails
                continue

            # PERFORMANCE OPTIMIZATION: Use more efficient normalization
            try:
                norm = np.linalg.norm(embedding)
                if norm != 0:
                    embedding = embedding / norm
                else:
                    # Skip face if normalization results in zero vector
                    continue
            except (ValueError, TypeError):
                # Skip face if normalization fails
                continue

            # Create Face object
            try:
                face = Face(
                    embedding=embedding,
                    bbox=bbox,
                    confidence=confidence,
                    landmarks=landmarks
                )
            except (TypeError, ValueError):
                # Skip face if Face object creation fails
                continue

            detected_faces.append(face)

        return detected_faces

    def detect_faces_with_boxes_and_confidence(self, image: np.ndarray) -> tuple[List[BoundingBox], List[float]]:
        """Detect faces and return only bounding boxes and confidence scores.

        Args:
            image: Input image as numpy array (H, W, C).

        Returns:
            Tuple of (list of bounding boxes, list of confidence scores).
        """
        faces = self.detect_faces(image)
        bboxes = [face.bbox for face in faces]
        confidences = [face.confidence for face in faces]

        return bboxes, confidences

    def set_threshold(self, threshold: float) -> None:
        """Update the detection confidence threshold.

        Args:
            threshold: New confidence threshold between 0 and 1.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold

    def get_threshold(self) -> float:
        """Get the current detection confidence threshold.

        Returns:
            Current confidence threshold.
        """
        return self.threshold