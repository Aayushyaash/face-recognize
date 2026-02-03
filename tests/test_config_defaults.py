"""Test script to verify AppConfig defaults."""

from face_recognize.config import AppConfig


def test_default_device_is_cpu() -> None:
    """Test that the default device in AppConfig is 'cpu'."""
    config = AppConfig()
    msg = f"Expected device to be 'cpu', but got '{config.device}'"
    assert config.device == "cpu", msg
    print("✅ AppConfig default device is correctly set to 'cpu'")


if __name__ == "__main__":
    test_default_device_is_cpu()

    print("All configuration tests passed!")
