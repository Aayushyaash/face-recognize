"""Face rendering module for visualization.

This module provides rendering of bounding boxes and identity labels
on video frames.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy.typing as npt

from ..config import AppConfig
from ..services.identification import IdentifiedFace


class FaceRenderer:
    """Renderer for drawing face bounding boxes and labels.

    Draws:
    - Green box + "Name (0.85)" for known faces
    - Red box + "Unknown" for unknown faces
    - White text with black outline for readability

    Attributes:
        config: Application configuration with colors and sizes.
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialize the renderer.

        Args:
            config: Application configuration with visualization settings.
        """
        self.config = config

    def render(
        self,
        frame: npt.NDArray[Any],
        faces: list[IdentifiedFace],
    ) -> npt.NDArray[Any]:
        """Render all faces on the frame.

        Modifies the frame in-place and also returns it.

        Args:
            frame: BGR image as numpy array, shape (H, W, 3).
            faces: List of identified faces to render.

        Returns:
            The same frame with annotations drawn.
        """
        for face in faces:
            self._draw_box(frame, face)
            self._draw_label(frame, face)

        return frame

    def _draw_box(self, frame: npt.NDArray[Any], face: IdentifiedFace) -> None:
        """Draw bounding box for a face.

        Args:
            frame: Image to draw on.
            face: Identified face with bbox and is_known.
        """
        # Select color based on known/unknown status
        color = self.config.known_color if face.is_known else self.config.unknown_color

        # Draw rectangle
        cv2.rectangle(
            frame,
            (face.bbox.x1, face.bbox.y1),
            (face.bbox.x2, face.bbox.y2),
            color,
            self.config.box_thickness,
        )

    def _draw_label(self, frame: npt.NDArray[Any], face: IdentifiedFace) -> None:
        """Draw identity label above bounding box.

        Format:
        - Known: "Name (0.85)"
        - Unknown: "Unknown"

        Args:
            frame: Image to draw on.
            face: Identified face with name, confidence, is_known.
        """
        # Format label text
        if face.is_known:
            label = f"{face.name} ({face.confidence:.2f})"
        else:
            label = "Unknown"

        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config.font_scale
        thickness = 1

        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Position label above bounding box
        # Ensure label doesn't go above frame
        label_y = max(face.bbox.y1 - 10, text_height + 5)
        label_x = face.bbox.x1

        # Draw black background rectangle for readability
        bg_pt1 = (label_x, label_y - text_height - 5)
        bg_pt2 = (label_x + text_width + 4, label_y + baseline - 5)

        # Select background color (darker version of box color)
        if face.is_known:
            bg_color = (0, 128, 0)  # Dark green
        else:
            bg_color = (0, 0, 128)  # Dark red

        cv2.rectangle(frame, bg_pt1, bg_pt2, bg_color, cv2.FILLED)

        # Draw text with outline for better visibility
        # First draw black outline
        cv2.putText(
            frame,
            label,
            (label_x + 2, label_y - 5),
            font,
            font_scale,
            (0, 0, 0),  # Black outline
            thickness + 1,
            cv2.LINE_AA,
        )

        # Then draw white text on top
        cv2.putText(
            frame,
            label,
            (label_x + 2, label_y - 5),
            font,
            font_scale,
            self.config.text_color,
            thickness,
            cv2.LINE_AA,
        )

    def render_fps(self, frame: npt.NDArray[Any], fps: float) -> None:
        """Render FPS counter in top-left corner.

        Args:
            frame: Image to draw on.
            fps: Current frames per second.
        """
        label = f"FPS: {fps:.1f}"
        cv2.putText(
            frame,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def render_status(self, frame: npt.NDArray[Any], message: str) -> None:
        """Render status message at bottom of frame.

        Args:
            frame: Image to draw on.
            message: Status message to display.
        """
        height = frame.shape[0]
        cv2.putText(
            frame,
            message,
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )