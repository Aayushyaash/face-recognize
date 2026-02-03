"""Camera abstraction for handling local and network video sources.

This module provides a unified interface for working with different camera types,
handling initialization, optimization (especially for network streams), and
robust frame reading with retry logic.
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy.typing as npt

from .logger import logger


class Camera:
    """Unified camera interface with robust handling and optimizations.

    Attributes:
        source: Camera index (int) or network URL (str).
        width: Desired frame width.
        height: Desired frame height.
        max_retries: Maximum number of consecutive read failures before giving up.
    """

    def __init__(
        self,
        source: int | str,
        width: int = 640,
        height: int = 480,
        max_retries: int = 5,
    ) -> None:
        """Initialize the camera.

        Args:
            source: Camera index or URL.
            width: Desired width.
            height: Desired height.
            max_retries: Max read retries.
        """
        self.source = source
        self.width = width
        self.height = height
        self.max_retries = max_retries
        self._cap: cv2.VideoCapture | None = None
        self._consecutive_failures = 0

    def __enter__(self) -> Camera:
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.release()

    def open(self) -> None:
        """Open the camera connection.

        Raises:
            RuntimeError: If connection fails.
        """
        if self._cap is not None and self._cap.isOpened():
            return

        logger.info(f"Connecting to camera source: {self.source}")

        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {self.source}")

        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Network stream optimization
        if isinstance(self.source, str):
            # Set buffer size to small to reduce latency on network streams
            # 38 is CAP_PROP_BUFFERSIZE
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            logger.debug("Network stream: Buffer size set to 1 for low latency")

        logger.info(f"Camera opened: {self.source}")

    def read(self) -> npt.NDArray[Any] | None:
        """Read a frame from the camera.

        Handles temporary failures by attempting to reconnect.

        Returns:
            Frame as numpy array, or None if failed after retries.
        """
        if self._cap is None or not self._cap.isOpened():
            try:
                self.open()
            except RuntimeError:
                return None

        if self._cap is None:
            return None

        ret, frame = self._cap.read()

        if ret:
            self._consecutive_failures = 0
            return frame

        # Handle failure
        self._consecutive_failures += 1
        logger.warning(
            f"Camera read failure ({self._consecutive_failures}/{self.max_retries})"
        )

        if self._consecutive_failures >= self.max_retries:
            logger.error("Max camera failures reached.")
            return None

        # Attempt reconnect
        self.release()
        time.sleep(1)  # Wait a bit before reconnecting
        try:
            self.open()
        except RuntimeError:
            pass  # Will try again on next read call or fail eventually

        return None

    def release(self) -> None:
        """Release camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.debug("Camera resources released")
