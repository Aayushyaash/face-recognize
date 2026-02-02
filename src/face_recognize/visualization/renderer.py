"""Face rendering module for visualization.

This module provides rendering of bounding boxes and identity labels
on video frames.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

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

        # Initialize font with fallback strategy
        try:
            # Try to load arial.ttf (Windows standard)
            font_obj = ImageFont.truetype("arial.ttf", size=20)
            self.font: ImageFont.ImageFont | ImageFont.FreeTypeFont = font_obj
        except OSError:
            # Fallback to default font
            self.font = ImageFont.load_default()

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

        # Convert BGR (OpenCV) -> RGB (PIL) only once per frame
        # We'll do this per label for now, but could optimize to do once per frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        # Calculate text size for background
        bbox = draw.textbbox((0, 0), label, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position label above bounding box
        # Ensure label doesn't go above frame
        label_y = max(face.bbox.y1 - 10, text_height + 5)
        label_x = face.bbox.x1

        # Draw black background rectangle for readability
        bg_pt1 = (label_x, label_y - text_height - 5)
        bg_pt2 = (label_x + text_width + 4, label_y + 5)

        # Select background color (darker version of box color)
        if face.is_known:
            bg_color = (0, 128, 0)  # Dark green
        else:
            bg_color = (0, 0, 128)  # Dark red

        draw.rectangle((*bg_pt1, *bg_pt2), fill=bg_color)

        # Draw text with outline for better visibility
        # First draw black outline
        outline_positions = [
            (label_x + 1, label_y - 5),
            (label_x + 3, label_y - 5),
            (label_x + 2, label_y - 6),
            (label_x + 2, label_y - 4),
        ]

        for pos in outline_positions:
            draw.text(pos, label, font=self.font, fill=(0, 0, 0))

        # Then draw white text on top
        draw.text(
            (label_x + 2, label_y - 5),
            label,
            font=self.font,
            fill=self.config.text_color,
        )

        # Convert RGB (PIL) -> BGR (OpenCV)
        frame[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def render_fps(self, frame: npt.NDArray[Any], fps: float) -> None:
        """Render FPS counter in top-left corner.

        Args:
            frame: Image to draw on.
            fps: Current frames per second.
        """
        label = f"FPS: {fps:.1f}"

        # Convert BGR (OpenCV) -> RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        # Draw text
        draw.text((10, 10), label, font=self.font, fill=(0, 255, 0))

        # Convert RGB (PIL) -> BGR (OpenCV)
        frame[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def render_status(self, frame: npt.NDArray[Any], message: str) -> None:
        """Render status message at bottom of frame.

        Args:
            frame: Image to draw on.
            message: Status message to display.
        """
        height = frame.shape[0]

        # Convert BGR (OpenCV) -> RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        # Draw text
        draw.text((10, height - 30), message, font=self.font, fill=(255, 255, 255))

        # Convert RGB (PIL) -> BGR (OpenCV)
        frame[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
