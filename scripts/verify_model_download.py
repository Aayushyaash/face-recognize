#!/usr/bin/env python3
"""
Model download verification script.

This script verifies that InsightFace models can be downloaded in the CI environment.
"""

import os
import sys
from pathlib import Path


def test_insightface_model_download() -> bool:
    """Test that InsightFace models can be downloaded."""
    try:
        import importlib.util

        if importlib.util.find_spec("insightface") is None:
            print("InsightFace not installed, skipping model download test")
            return True

        from insightface.app import FaceAnalysis

        # Initialize FaceAnalysis with the default buffalo_s model
        print("Initializing FaceAnalysis with buffalo_s model...")

        # Temporarily redirect models directory to avoid permission issues in CI
        old_home = os.environ.get("HOME")
        temp_home = Path(".temp_home")
        temp_home.mkdir(exist_ok=True)
        os.environ["HOME"] = str(temp_home.absolute())

        # Try to initialize FaceAnalysis which will trigger model download
        app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))

        # Restore original HOME if it existed
        if old_home is not None:
            os.environ["HOME"] = old_home
        else:
            del os.environ["HOME"]

        print("Successfully initialized FaceAnalysis with buffalo_s model")
        return True
    except ImportError:
        print("InsightFace not installed, skipping model download test")
        return True  # Not an error in CI if not installed yet
    except Exception as e:
        print(f"Error downloading InsightFace model: {str(e)}")
        return False


def main() -> int:
    """Main function to run model download verification."""
    print("Verifying InsightFace model download capability...")

    success = test_insightface_model_download()

    if success:
        print("Model download verification passed!")
        return 0
    else:
        print("Model download verification failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
