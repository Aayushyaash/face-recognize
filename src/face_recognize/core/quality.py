"""Face quality assessment module for evaluating face image quality.

This module implements a FaceQualityScorer that evaluates face image quality
based on sharpness and brightness metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from face_recognize.core.models import BoundingBox


class FaceQualityScorer:
    """Evaluates face image quality based on sharpness and brightness.

    The quality score is a combination of sharpness (blur detection) and
    brightness (mean pixel intensity) metrics, normalized to a 0.0-1.0 range.
    """

    def __init__(
        self, sharpness_weight: float = 0.6, brightness_weight: float = 0.4
    ) -> None:
        """Initialize the quality scorer with configurable weights.

        Args:
            sharpness_weight: Weight for sharpness score in composite score (0-1).
            brightness_weight: Weight for brightness score in composite score (0-1).
                Note: sharpness_weight + brightness_weight should equal 1.0.
        """
        self.sharpness_weight = sharpness_weight
        self.brightness_weight = brightness_weight
        # Normalize weights to sum to 1.0
        total_weight = sharpness_weight + brightness_weight
        if total_weight != 0:
            self.sharpness_weight /= total_weight
            self.brightness_weight /= total_weight

    def evaluate(
        self, image: npt.NDArray[np.uint8], bbox: BoundingBox | None = None
    ) -> float:
        """Evaluate the quality of a face in an image.

        Args:
            image: Input image containing the face (BGR format).
            bbox: Optional bounding box to crop the face region.
                If None, evaluates the entire image.

        Returns:
            Quality score between 0.0 (low quality) and 1.0 (high quality).
        """
        # Crop the face region if bbox is provided
        if bbox is not None:
            face_region = image[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
        else:
            face_region = image

        # Ensure we have a valid face region
        if face_region.size == 0:
            return 0.0

        # Calculate individual metrics
        sharpness_score = self._calculate_sharpness(face_region)
        brightness_score = self._calculate_brightness(face_region)

        # Combine scores using weighted average
        quality_score = (
            self.sharpness_weight * sharpness_score
            + self.brightness_weight * brightness_score
        )

        return quality_score

    def _calculate_sharpness(self, image: npt.NDArray[np.uint8]) -> float:
        """Calculate sharpness score using Laplacian variance.

        Args:
            image: Input image region (BGR format).

        Returns:
            Sharpness score between 0.0 (blurred) and 1.0 (sharp).
        """
        # Convert to grayscale for sharpness calculation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate Laplacian variance - higher values indicate sharper images
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Normalize the sharpness score to 0-1 range
        # Using logarithmic scaling to handle wide range of values
        if laplacian_var <= 0:
            return 0.0

        # Normalize using a logarithmic scale with a reasonable upper bound
        # This assumes that very sharp images might have Laplacian variance up to ~1000
        normalized_score = min(laplacian_var / 100.0, 1.0)  # Adjust divisor as needed
        return min(normalized_score, 1.0)

    def _calculate_brightness(self, image: npt.NDArray[np.uint8]) -> float:
        """Calculate brightness score based on mean pixel intensity.

        Args:
            image: Input image region (BGR format).

        Returns:
            Brightness score between 0.0 (too dark) and 1.0 (optimal brightness).
        """
        # Convert to grayscale for brightness calculation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate mean brightness
        mean_brightness = float(np.mean(gray.astype(np.float64)))

        # Normalize brightness to 0-1 range
        # Optimal brightness is around middle of the range (128 for 0-255)
        # Using a Gaussian-like curve centered at 128
        optimal_brightness = 128.0
        std_dev = 64.0  # Controls how quickly score drops off from optimal

        # Calculate normalized score using Gaussian function
        exponent = -0.5 * ((mean_brightness - optimal_brightness) / std_dev) ** 2
        brightness_score = float(np.exp(exponent))

        return brightness_score


# Default quality scorer instance
DEFAULT_QUALITY_SCORER = FaceQualityScorer()
