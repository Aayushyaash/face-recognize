"""Threaded camera implementation for multi-camera support."""

from __future__ import annotations

import queue
import threading
import time

import cv2
import numpy as np
import numpy.typing as npt

from ..core.logger import logger


class ThreadedCamera:
    """Threaded camera wrapper for multi-camera support with frame buffering.

    This class wraps OpenCV's VideoCapture in a separate thread to allow
    concurrent access to multiple cameras without blocking. It uses a queue
    with maxsize=1 to implement single frame buffering (dropping older frames
    when new ones arrive).
    """

    def __init__(
        self,
        source: int | str,
        frame_width: int = 640,
        frame_height: int = 480,
        buffer_size: int = 1,
    ) -> None:
        """Initialize the threaded camera.

        Args:
            source: Camera index (0, 1, ...) or URL string.
            frame_width: Desired frame width.
            frame_height: Desired frame height.
            buffer_size: Size of frame buffer (default 1 for single frame buffering).
        """
        self.source = source
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.buffer_size = buffer_size

        # Initialize camera
        self.cap = cv2.VideoCapture()
        self._configure_camera()

        # Frame buffer queue
        self.frame_queue: queue.Queue[npt.NDArray[np.uint8] | None] = queue.Queue(
            maxsize=buffer_size
        )

        # Threading controls
        self.running = False
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()

    def _configure_camera(self) -> None:
        """Configure the camera properties."""
        if not self.cap.isOpened():
            self.cap.open(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera: {self.source}")

        # Set properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        # Optional: Set buffer size to 1 to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def start(self) -> None:
        """Start the camera thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the camera thread."""
        if not self.running:
            return

        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)  # Wait up to 2 seconds for thread to finish

        # Release camera
        with self.lock:
            if self.cap.isOpened():
                self.cap.release()

    def read(self) -> npt.NDArray[np.uint8] | None:
        """Read the latest frame from the buffer.

        Returns:
            Latest frame if available, None if no frames or camera stopped.
        """
        try:
            # Get the latest frame from the queue
            frame = self.frame_queue.get_nowait()
            return frame
        except queue.Empty:
            return None

    def _capture_loop(self) -> None:
        """Internal capture loop running in separate thread."""
        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                # Log error but continue trying to capture
                logger.warning(f"Failed to read frame from camera: {self.source}")
                time.sleep(0.01)  # Brief pause before retrying
                continue

            # Put frame in queue, dropping oldest if buffer full
            try:
                # If queue is full, remove the oldest frame first
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()  # Remove oldest frame
                    except queue.Empty:
                        pass  # Queue became empty between full() and get_nowait()

                # Add new frame
                self.frame_queue.put_nowait(frame.astype(np.uint8))
            except queue.Full:
                # This shouldn't happen due to the check above, but just in case
                pass

            # Small delay to prevent excessive CPU usage
            time.sleep(0.001)

    def is_opened(self) -> bool:
        """Check if the camera is opened.

        Returns:
            True if camera is opened, False otherwise.
        """
        with self.lock:
            result: bool = self.cap.isOpened()
            return result

    def get_property(self, prop_id: int) -> float:
        """Get a property value from the camera.

        Args:
            prop_id: Property ID (e.g., cv2.CAP_PROP_FRAME_WIDTH).

        Returns:
            Property value.
        """
        with self.lock:
            result: float = self.cap.get(prop_id)
            return result

    def set_property(self, prop_id: int, value: float) -> bool:
        """Set a property value for the camera.

        Args:
            prop_id: Property ID (e.g., cv2.CAP_PROP_FRAME_WIDTH).
            value: Property value to set.

        Returns:
            True if successful, False otherwise.
        """
        with self.lock:
            result: bool = self.cap.set(prop_id, value)
            return result
