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
# Clone repository
git clone https://github.com/yourusername/face-recognize.git
cd face-recognize

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install uv (recommended package manager)
pip install uv

# Install in development mode using uv
uv pip install -e ".[dev]"
```

> **Note:** This is currently a development/personal project. PyPI package not yet published.

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

### Using the Face Detection Module

```python
from src.face_recognize.core.detector import FaceDetector
from src.face_recognize.config import DEFAULT_CONFIG
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
face-recognize run [--camera 0] [--model buffalo_s] [--threshold 0.4]
```

Starts real-time face identification from the camera feed.

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
- **Visualization Layer**: Draws bounding boxes and labels on the video feed

For implementation details, see the Architecture section above.

---

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/face-recognize.git
cd face-recognize

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install development dependencies using uv (recommended)
uv pip install -e ".[dev]"

# Alternative: Install using pip
# pip install -e ".[dev]"
```

### Code Quality

```bash
# Format code
uv run black src/ tests/

# Lint
uv run ruff check src/

# Type check
uv run mypy src/

# Run tests
uv run pytest tests/ -v
```
## License

MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) - Face detection and recognition models
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - Efficient model inference
- [OpenCV](https://opencv.org/) - Computer vision operations