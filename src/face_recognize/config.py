"""Application configuration."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Application configuration with sensible defaults.

    Attributes:
        model: InsightFace model pack name (buffalo_s, buffalo_l, buffalo_sc).
        device: Inference device ('cpu' or 'cuda').
        detection_threshold: Minimum detection confidence (0-1).
        max_track_age: Frames before removing unmatched track.
        iou_threshold: Minimum IoU for bounding box matching.
        similarity_threshold: Minimum embedding similarity for match.
        unknown_cache_ttl: Seconds to cache unknown identity.
        min_quality_score: Minimum quality score for registration.
        database_path: Path to JSON database file.
        camera_index: Camera device index.
        frame_width: Camera frame width.
        frame_height: Camera frame height.
    """

    # Model settings
    model: str = "buffalo_s"
    device: str = "cpu"

    # Detection settings
    detection_threshold: float = 0.7  # Updated to 0.7 as per requirements

    # Tracking settings
    max_track_age: int = 30
    iou_threshold: float = 0.3
    similarity_threshold: float = 0.4

    # Identification settings
    unknown_cache_ttl: float = 30.0

    # Registration settings
    min_quality_score: float = 0.5

    # Database settings
    database_path: Path = field(default_factory=lambda: Path("data/faces.json"))

    # Camera settings
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480

    # Visualization settings
    known_color: tuple[int, int, int] = (0, 255, 0)  # Green BGR
    unknown_color: tuple[int, int, int] = (0, 0, 255)  # Red BGR
    text_color: tuple[int, int, int] = (255, 255, 255)  # White BGR
    box_thickness: int = 2
    font_scale: float = 0.6


# Default configuration instance
DEFAULT_CONFIG = AppConfig()
