"""Unit tests for the Camera class."""

from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from face_recognize.core.camera import Camera


@pytest.fixture
def mock_cv2() -> Generator[MagicMock, None, None]:
    """Mock the cv2 module."""
    with patch("face_recognize.core.camera.cv2") as mock:
        # constant values
        mock.CAP_PROP_FRAME_WIDTH = 3
        mock.CAP_PROP_FRAME_HEIGHT = 4
        mock.CAP_PROP_BUFFERSIZE = 38
        mock.CAP_ANY = 0
        mock.CAP_DSHOW = 700
        yield mock


def test_camera_init(mock_cv2: MagicMock) -> None:
    """Test camera initialization."""
    cam = Camera(0)
    assert cam.source == 0
    assert cam.width == 640
    assert cam.height == 480

    cam_url = Camera("http://example.com/video")
    assert cam_url.source == "http://example.com/video"


def test_camera_open_success(mock_cv2: MagicMock) -> None:
    """Test successful camera opening."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cv2.VideoCapture.return_value = mock_cap

    with Camera(0) as cam:
        assert cam._cap is not None
        mock_cv2.VideoCapture.assert_called_with(0)
        # Check resolution setup
        mock_cap.set.assert_any_call(mock_cv2.CAP_PROP_FRAME_WIDTH, 640)
        mock_cap.set.assert_any_call(mock_cv2.CAP_PROP_FRAME_HEIGHT, 480)


def test_camera_open_network_optimization(mock_cv2: MagicMock) -> None:
    """Test network stream optimization settings."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cv2.VideoCapture.return_value = mock_cap

    with Camera("http://url"):
        mock_cv2.VideoCapture.assert_called_with("http://url")
        # Check buffer size optimization for network stream
        mock_cap.set.assert_any_call(mock_cv2.CAP_PROP_BUFFERSIZE, 1)


def test_camera_read_success(mock_cv2: MagicMock) -> None:
    """Test successful frame reading."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, frame)
    mock_cv2.VideoCapture.return_value = mock_cap

    cam = Camera(0)
    cam.open()
    result = cam.read()

    assert result is not None
    assert np.array_equal(result, frame)
    assert cam._consecutive_failures == 0


def test_camera_reconnect_logic(mock_cv2: MagicMock) -> None:
    """Test retry and reconnect logic."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Sequence: Fail, Fail, Success
    mock_cap.read.side_effect = [(False, None), (False, None), (True, frame)]
    mock_cv2.VideoCapture.return_value = mock_cap

    cam = Camera(0, max_retries=3)
    cam.open()

    # First read fails
    res1 = cam.read()
    assert res1 is None
    assert cam._consecutive_failures == 1

    # Second read fails
    res2 = cam.read()
    assert res2 is None
    assert cam._consecutive_failures == 2

    # Third read succeeds
    res3 = cam.read()
    assert res3 is not None
    assert cam._consecutive_failures == 0
