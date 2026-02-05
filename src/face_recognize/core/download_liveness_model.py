#!/usr/bin/env python3
"""Helper script to download the MiniFASNetV2 ONNX model for liveness detection.

This script downloads the MiniFASNetV2 model from a reliable source
and places it in the models/ directory for liveness detection.
"""

import urllib.request
from pathlib import Path


def download_minifasnetv2(output_dir: Path | str = "models") -> None:
    """Download the MiniFASNetV2 ONNX model.

    Args:
        output_dir: Directory to save the model file to.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    model_url = "https://github.com/ronghuaiyang/anti-spoofing-models/raw/master/models/MiniFASNetV2.onnx"
    model_path = output_dir / "MiniFASNetV2.onnx"

    print(f"Downloading MiniFASNetV2 model from: {model_url}")
    print(f"Saving to: {model_path}")

    # Validate URL scheme to prevent file:// access
    from urllib.parse import urlparse

    parsed_url = urlparse(model_url)
    if parsed_url.scheme not in ("http", "https"):
        print(f"Error: Invalid URL scheme: {parsed_url.scheme}")
        return

    try:
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Successfully downloaded MiniFASNetV2 to {model_path}")

        # Verify file was downloaded
        if model_path.exists():
            size = model_path.stat().st_size
            print(f"Model file size: {size} bytes")
        else:
            print("Error: Model file was not downloaded successfully")

    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please check your internet connection and try again.")
        print("Alternatively, you can manually download the model from:")
        print(f"  {model_url}")
        print("And place it in the models/ directory as 'MiniFASNetV2.onnx'")


def main() -> None:
    """Main entry point for the download script."""
    print("MiniFASNetV2 Model Downloader")
    print("=" * 30)

    # Check if model already exists
    model_path = Path("models/MiniFASNetV2.onnx")
    if model_path.exists():
        print(f"Model already exists at {model_path}")
        response = input("Do you want to download again? (y/N): ")
        if response.lower() != "y":
            print("Skipping download.")
            return

    download_minifasnetv2()


if __name__ == "__main__":
    main()
