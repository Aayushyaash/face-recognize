"""Unit tests for updated AppConfig with quality and liveness parameters."""

from pathlib import Path

from face_recognize.config import AppConfig


class TestAppConfigUpdated:
    """Test cases for updated AppConfig with new parameters."""

    def test_default_values_for_new_parameters(self) -> None:
        """Test that new parameters have correct default values."""
        config = AppConfig()

        # Test new liveness parameters
        assert config.liveness_threshold == 0.5
        assert config.enable_liveness is True

        # Test updated min_quality_score
        assert config.min_quality_score == 0.6  # Updated from 0.5

        # Test new database backend parameter
        assert config.database_backend == "json"  # Default to JSON backend

    def test_custom_values_for_new_parameters(self) -> None:
        """Test that new parameters can be set to custom values."""
        config = AppConfig(
            liveness_threshold=0.7,
            enable_liveness=False,
            min_quality_score=0.8,
            database_backend="sqlite",
        )

        assert config.liveness_threshold == 0.7
        assert config.enable_liveness is False
        assert config.min_quality_score == 0.8
        assert config.database_backend == "sqlite"

    def test_backward_compatibility(self) -> None:
        """Test that existing parameters still work as expected."""
        config = AppConfig(
            detection_threshold=0.8, similarity_threshold=0.5, camera_index=1
        )

        assert config.detection_threshold == 0.8
        assert config.similarity_threshold == 0.5
        assert config.camera_index == 1
        # Ensure new parameters still have correct defaults
        assert config.liveness_threshold == 0.5
        assert config.enable_liveness is True
        assert config.database_backend == "json"

    def test_database_path_default(self) -> None:
        """Test that database path default is preserved."""
        config = AppConfig()
        assert config.database_path == Path("data/faces.json")
