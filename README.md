# Face-Recognize

> A standalone real-time face identification system that detects faces in a live camera feed, extracts facial embeddings using InsightFace, matches them against a local vector database, and displays the identified person's name overlaid on the video feed.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Features

- **🎯 Real-time Identification**: Recognize known individuals in live camera feed
- **👤 Face Registration**: Add new faces to the local database via CLI
- **📋 Database Management**: List, view, and delete registered persons
- **🎨 Visual Feedback**: Green boxes for known faces, red for unknown
- **⚡ Fast Performance**: Optimized for real-time processing
- **🔒 Privacy Focused**: All processing happens locally
- **🔌 Extensible Design**: Modular architecture for easy customization

---

## Installation

```bash
# Standard Install (CPU Default)
uv pip install -e .

# Development Install
uv pip install -e ".[dev]"
```

> **Note:** This is a development/personal project.

## GPU Support

### Prerequisites
- NVIDIA GPU with compute capability 6.0+
- NVIDIA CUDA 12.x
- cuDNN 9.x

### Activation
If you want to enable GPU support, you must manually install the GPU packages:

```bash
# 1. Uninstall CPU versions
uv pip uninstall onnxruntime opencv-python-headless

# 2. Install GPU version
uv pip install onnxruntime-gpu
```

> **Note:** If you encounter errors about `cv2.imshow`, ensure `opencv-python-headless` is NOT installed.

---

## Quick Start

### Register a Person

```bash
# Register a new person from an image
face-recognize register path/to/photo.jpg "John Doe"
```

### List Registered Persons

```bash
# List all registered persons
face-recognize list
```

### Start Real-time Identification

```bash
# Start the camera identification system
face-recognize run
```

Press 'q' to quit the camera window.

### Using IP Cameras / WiFi Streams

To connect to a network camera (including "IP Webcam" Android app):

```bash
# Connect to an IP camera (RTSP or HTTP)
face-recognize run --camera "http://192.168.1.100:8080/video"

# Connect to RTSP stream
face-recognize run --camera "rtsp://user:password@192.168.1.100:554/stream"
```

**Note for "IP Webcam" App Users:**  
1. Install "IP Webcam" by Pavel Khlebovich (Thyoni Tech).
2. Start the server on your phone.
3. Use the URL format `http://<PHONE_IP>:8080/video`.

### Using the Face Detection Module

```python
from face_recognize.core.detector import FaceDetector
from face_recognize.config import DEFAULT_CONFIG
import cv2

# Initialize the face detector
detector = FaceDetector(config=DEFAULT_CONFIG)

# Load an image
image = cv2.imread("path/to/image.jpg")

# Detect faces
faces = detector.detect_faces(image)

# Print results
for i, face in enumerate(faces):
    print(f"Face {i+1}:")
    print(f"  Bounding Box: ({face.bbox.x1}, {face.bbox.y1}) to ({face.bbox.x2}, {face.bbox.y2})")
    print(f"  Confidence: {face.confidence:.2f}")
    print(f"  Embedding shape: {face.embedding.shape}")
```

### Changing Detection Threshold

```python
# Get current threshold
current_threshold = detector.get_threshold()
print(f"Current threshold: {current_threshold}")

# Set new threshold
detector.set_threshold(0.8)
print(f"New threshold: {detector.get_threshold()}")
```

### Using Different Models

```python
# Change to a different InsightFace model
detector.change_model('buffalo_l')  # or 'buffalo_s', 'buffalo_sc'
```

---

## CLI Commands

### `run` - Start Camera Identification

```bash
face-recognize run [--camera 0] [--model buffalo_s] [--threshold 0.4] [--device cpu|cuda]
```

Starts real-time face identification from the camera feed.

Options:
- `--device <cpu|cuda>`: Specify inference device (Default: cpu)

### `register` - Add New Person

```bash
face-recognize register <image_path> <name>
```

Registers a new person in the database from an image file.

### `list` - Show All Persons

```bash
face-recognize list
```

Lists all registered persons in the database.

### `delete` - Remove Person

```bash
face-recognize delete <name>
```

Removes a person from the database.

### `info` - View Person Details

```bash
face-recognize info <name>
```

Shows details about a registered person.

---

## Architecture

The system consists of several key components:

- **Face Detector**: Uses InsightFace to detect faces and extract embeddings
- **Face Tracker**: Maintains persistent IDs across video frames using IoU matching
- **Identification Service**: Matches embeddings against the local database
- **Database Backend**: Stores registered faces in JSON format
  - **PersonRecord**: Data class for storing face records with ID, name, embedding, and timestamp
  - **JsonDatabase**: Thread-safe database with atomic write operations, supporting CRUD operations and similarity search
- **Visualization Layer**: Draws bounding boxes and labels on the video feed

For implementation details, see the Architecture section above.

### Project Structure

```text
face-recognize/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions for CI/CD (linting, testing, security)
├── .vscode/
│   └── tasks.json               # VS Code tasks for development automation
├── docs/                        # Documentation and planning artifacts
│   └── ...                      (Implementation plans and reports)
├── src/
│   └── face_recognize/          # Main application package
│       ├── cli/
│       │   ├── commands.py      # Implementation of CLI commands (run, register, etc.)
│       │   └── main.py          # Entry point and argument parsing
│       ├── core/                # Core business logic
│       │   ├── camera.py        # Camera handling and frame capture
│       │   ├── detector.py      # Face detection using InsightFace/ONNX
│       │   ├── logger.py        # Logging configuration
│       │   └── models.py        # Core data models (BoundingBox, Face, etc.)
│       ├── database/            # Data persistence layer
│       │   ├── json_db.py       # JSON-based storage implementation
│       │   └── models.py        # Database record models
│       ├── services/            # Higher-level services
│       │   ├── identification.py # Matching embeddings against database
│       │   └── tracker.py       # Object tracking across frames
│       ├── visualization/       # UI rendering
│       │   └── drawer.py        # Drawing bounding boxes and labels
│       ├── config.py            # Global application configuration
│       └── __init__.py
├── tests/                       # Test suite
│   ├── test_config_defaults.py  # Configuration unit tests
│   └── ...                      (Other unit and integration tests)
├── .pre-commit-config.yaml      # Git hooks configuration (linting before commit)
├── LICENSE                      # MIT License file
├── pyproject.toml               # Project metadata and dependencies
├── README.md                    # Project documentation
└── uv.lock                      # Exact versions of installed dependencies
```

---

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/Aayushyaash/face-recognize.git
cd face-recognize

# Create and activate virtual environment
# Using uv (Recommended)
uv venv

# Alternative: Standard Python
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
uv sync

# Install development dependencies
uv sync --dev
```

### Code Quality

```bash
# Format code
uv run ruff format src/ tests/

# Lint
uv run ruff check src/

# Type check
uv run mypy src/

# Run tests with coverage
uv run pytest tests/ --cov=src/ --cov-report=term-missing

# Security Scan
uv run pip install bandit pip-audit && uv run bandit -r src/ && uv run pip-audit .
```
## License

MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) - Face detection and recognition models
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - Efficient model inference
- [OpenCV](https://opencv.org/) - Computer vision operations