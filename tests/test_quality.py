"""Unit tests for FaceQualityScorer."""

import numpy as np

from face_recognize.core.models import BoundingBox
from face_recognize.core.quality import FaceQualityScorer


class TestFaceQualityScorer:
    """Test cases for FaceQualityScorer class."""

    def test_initialization_with_weights(self) -> None:
        """Test initialization with custom weights."""
        scorer = FaceQualityScorer(sharpness_weight=0.7, brightness_weight=0.3)

        assert scorer.sharpness_weight == 0.7
        assert scorer.brightness_weight == 0.3

    def test_weight_normalization(self) -> None:
        """Test that weights are normalized to sum to 1.0."""
        scorer = FaceQualityScorer(sharpness_weight=0.6, brightness_weight=0.4)

        # Total should be 1.0 after normalization
        total_weight = scorer.sharpness_weight + scorer.brightness_weight
        assert abs(total_weight - 1.0) < 1e-10

    def test_evaluate_with_bbox(self) -> None:
        """Test quality evaluation with bounding box."""
        # Create a test image with a bright, sharp region
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[20:80, 20:80] = [200, 200, 200]  # Bright region in the center

        # Add some detail to make it sharp
        image[40:60, 40:60] = [50, 50, 50]  # Darker center square

        bbox = BoundingBox(x1=20, y1=20, x2=80, y2=80)
        scorer = FaceQualityScorer()

        quality_score = scorer.evaluate(image, bbox)

        # The score should be reasonably high for a bright, sharp region
        assert 0.0 <= quality_score <= 1.0

    def test_evaluate_without_bbox(self) -> None:
        """Test quality evaluation on entire image."""
        # Create a test image with mixed quality regions
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[20:80, 20:80] = [200, 200, 200]  # Bright region in the center
        image[0:20, 0:20] = [10, 10, 10]  # Dark corner

        scorer = FaceQualityScorer()

        quality_score = scorer.evaluate(image)

        # The score should be between 0 and 1
        assert 0.0 <= quality_score <= 1.0

    def test_evaluate_empty_region(self) -> None:
        """Test quality evaluation with empty face region."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Invalid bbox (x2 < x1, y2 < y1)
        bbox = BoundingBox(x1=50, y1=50, x2=40, y2=40)

        scorer = FaceQualityScorer()

        quality_score = scorer.evaluate(image, bbox)

        # Should return 0.0 for invalid/empty region
        assert quality_score == 0.0

    def test_calculate_sharpness_sharp_image(self) -> None:
        """Test sharpness calculation for a sharp image."""
        # Create a sharp image with clear edges
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        image[0:25, :] = [255, 255, 255]  # White top half
        image[25:50, :] = [0, 0, 0]  # Black bottom half

        scorer = FaceQualityScorer()
        sharpness_score = scorer._calculate_sharpness(image)

        # Should have high sharpness due to clear edge
        assert 0.0 <= sharpness_score <= 1.0
        # The score should be relatively high for a sharp image
        assert sharpness_score > 0.1

    def test_calculate_sharpness_blurry_image(self) -> None:
        """Test sharpness calculation for a blurry image."""
        # Create a smooth gradient (blurry)
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        for i in range(50):
            image[:, i] = [i * 5, i * 5, i * 5]  # Smooth gradient

        scorer = FaceQualityScorer()
        sharpness_score = scorer._calculate_sharpness(image)

        # Should have low sharpness due to lack of edges
        assert 0.0 <= sharpness_score <= 1.0

    def test_calculate_brightness_optimal(self) -> None:
        """Test brightness calculation for optimally bright image."""
        # Create an image with optimal brightness (around 128)
        image = np.full((50, 50, 3), (128, 128, 128), dtype=np.uint8)

        scorer = FaceQualityScorer()
        brightness_score = scorer._calculate_brightness(image)

        # Should have high brightness score for optimal brightness
        assert 0.0 <= brightness_score <= 1.0
        # Should be close to 1.0 for optimal brightness
        assert brightness_score > 0.8

    def test_calculate_brightness_dark(self) -> None:
        """Test brightness calculation for dark image."""
        # Create a dark image
        image = np.full((50, 50, 3), (20, 20, 20), dtype=np.uint8)

        scorer = FaceQualityScorer()
        brightness_score = scorer._calculate_brightness(image)

        # Should have lower brightness score for dark image
        assert 0.0 <= brightness_score <= 1.0

    def test_calculate_brightness_bright(self) -> None:
        """Test brightness calculation for bright image."""
        # Create a bright image
        image = np.full((50, 50, 3), (230, 230, 230), dtype=np.uint8)

        scorer = FaceQualityScorer()
        brightness_score = scorer._calculate_brightness(image)

        # Should have moderate score for very bright image (not optimal)
        assert 0.0 <= brightness_score <= 1.0

    def test_quality_score_combination(self) -> None:
        """Test that quality score combines sharpness and brightness."""
        # Create images with different characteristics
        sharp_image = np.zeros((50, 50, 3), dtype=np.uint8)
        sharp_image[0:25, :] = [255, 255, 255]  # Sharp edge
        sharp_image[25:50, :] = [0, 0, 0]

        bright_image = np.full((50, 50, 3), (128, 128, 128), dtype=np.uint8)

        # Combine both characteristics
        combined_image = np.copy(sharp_image)
        combined_image[10:40, 10:40] = bright_image[10:40, 10:40]

        scorer = FaceQualityScorer(sharpness_weight=0.5, brightness_weight=0.5)
        quality_score = scorer.evaluate(combined_image)

        assert 0.0 <= quality_score <= 1.0

        # Test with different weight combinations
        scorer_heavy_sharpness = FaceQualityScorer(
            sharpness_weight=0.8, brightness_weight=0.2
        )
        score_heavy_sharpness = scorer_heavy_sharpness.evaluate(combined_image)

        assert 0.0 <= score_heavy_sharpness <= 1.0
