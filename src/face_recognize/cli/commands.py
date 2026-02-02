"""Command implementations for the face-recognize CLI.

Each command function takes parsed args and config, returns exit code.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2

from ..config import AppConfig
from ..core.detector import FaceDetector
from ..core.tracker import FaceTracker
from ..database.json_backend import JsonDatabase
from ..services.identification import IdentificationService
from ..visualization.renderer import FaceRenderer


def cmd_run(args: argparse.Namespace, config: AppConfig) -> int:
    """Run real-time camera identification.

    Args:
        args: Parsed arguments with camera, model, threshold.
        config: Application configuration.

    Returns:
        Exit code (0 for success).
    """
    # Apply command-line overrides to config
    config = replace(
        config,
        camera_index=args.camera,
        model=args.model,
        similarity_threshold=args.threshold,
    )
    print(f"Initializing with model: {config.model}")
    print(f"Similarity threshold: {config.similarity_threshold}")

    # Initialize components
    try:
        detector = FaceDetector(config)
    except Exception as e:
        print(f"Error initializing face detector: {e}", file=sys.stderr)
        print("Make sure the model is downloaded to ./models/", file=sys.stderr)
        return 1

    tracker = FaceTracker(config)
    database = JsonDatabase(config.database_path)
    identifier = IdentificationService(database, config)
    renderer = FaceRenderer(config)

    print(f"Database loaded: {database.count()} persons")

    # Open camera
    backend = cv2.CAP_ANY
    if platform.system() == "Windows":
        backend = cv2.CAP_DSHOW

    cap = cv2.VideoCapture(config.camera_index, backend)
    if not cap.isOpened():
        # Fallback to default if DSHOW fails
        cap = cv2.VideoCapture(config.camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera {config.camera_index}", file=sys.stderr)
        return 1

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

    print(f"Camera {config.camera_index} opened. Press 'q' to quit.")

    # FPS calculation
    frame_times: list[float] = []
    fps = 0.0

    # Camera failure handling
    max_retries = 5
    consecutive_failures = 0

    try:
        while True:
            frame_start = time.time()

            # Capture frame
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                msg = f"Warning: Camera read failure ({consecutive_failures}/"
                msg += f"{max_retries})"
                print(msg, file=sys.stderr)

                if consecutive_failures >= max_retries:
                    print(
                        "Error: Max camera failures reached. Exiting.", file=sys.stderr
                    )
                    break

                # Attempt to reconnect
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(config.camera_index, backend)
                if not cap.isOpened():
                    # Fallback to default if DSHOW fails
                    cap = cv2.VideoCapture(config.camera_index)

                if not cap.isOpened():
                    print(
                        f"Error: Could not reopen camera {config.camera_index}",
                        file=sys.stderr,
                    )
                    break

                # Set camera resolution again after reopening
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

                continue

            # successful read
            consecutive_failures = 0

            # Detect faces
            faces = detector.detect_faces(frame)

            # Track faces
            tracked_faces = tracker.update(faces)

            # Identify faces
            identified_faces = identifier.identify(tracked_faces)

            # Render
            renderer.render(frame, identified_faces)
            renderer.render_fps(frame, fps)

            # Display
            cv2.imshow("Face-Recognize", frame)

            # Calculate FPS
            frame_end = time.time()
            frame_times.append(frame_end - frame_start)
            if len(frame_times) > 30:
                frame_times.pop(0)
            if frame_times:
                fps = 1.0 / (sum(frame_times) / len(frame_times))

            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


def cmd_register(args: argparse.Namespace, config: AppConfig) -> int:
    """Register a new person from an image.

    Args:
        args: Parsed arguments with image_path, name.
        config: Application configuration.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    image_path: Path = args.image_path
    name: str = args.name.strip()

    # Validate inputs
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}", file=sys.stderr)
        return 1

    if not name:
        print("Error: Name cannot be empty", file=sys.stderr)
        return 1

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image: {image_path}", file=sys.stderr)
        return 1

    print(f"Processing image: {image_path}")

    # Initialize detector
    try:
        detector = FaceDetector(config)
    except Exception as e:
        print(f"Error initializing detector: {e}", file=sys.stderr)
        return 1

    # Detect faces
    faces = detector.detect_faces(image)

    # Validate exactly one face
    if len(faces) == 0:
        print("Error: No face detected in image", file=sys.stderr)
        return 1

    if len(faces) > 1:
        print(
            f"Error: Multiple faces detected ({len(faces)}). "
            "Please use an image with exactly one face.",
            file=sys.stderr,
        )
        return 1

    face = faces[0]

    # Validate face quality (using confidence as proxy)
    if face.confidence < config.min_quality_score:
        print(
            f"Error: Face quality too low ({face.confidence:.2f}). "
            f"Minimum required: {config.min_quality_score}",
            file=sys.stderr,
        )
        return 1

    # Add to database
    database = JsonDatabase(config.database_path)

    try:
        record = database.add(name, face.embedding)
        print(f'✓ Registered "{name}" successfully (ID: {record.id[:8]})')
        return 0
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_list(args: argparse.Namespace, config: AppConfig) -> int:
    """List all registered persons.

    Args:
        args: Parsed arguments (unused).
        config: Application configuration.

    Returns:
        Exit code (0 for success).
    """
    database = JsonDatabase(config.database_path)
    persons = database.list_all()

    if not persons:
        print("No registered persons.")
        return 0

    print(f"Registered Persons ({len(persons)}):")
    for i, person in enumerate(persons, 1):
        # Parse and format timestamp
        try:
            # ISO format: 2026-01-30T10:15:23.456789
            timestamp = person.created_at.split("T")
            date_part = timestamp[0]
            time_part = timestamp[1].split(".")[0] if len(timestamp) > 1 else ""
            formatted_time = f"{date_part} {time_part}"
        except (IndexError, ValueError):
            formatted_time = person.created_at

        print(f"  {i}. {person.name:<20} (registered: {formatted_time})")

    return 0


def cmd_delete(args: argparse.Namespace, config: AppConfig) -> int:
    """Delete a person from the database.

    Args:
        args: Parsed arguments with name.
        config: Application configuration.

    Returns:
        Exit code (0 for success, 1 if not found).
    """
    name: str = args.name.strip()

    if not name:
        print("Error: Name cannot be empty", file=sys.stderr)
        return 1

    database = JsonDatabase(config.database_path)

    if database.delete(name):
        print(f'✓ Deleted "{name}" from database')
        return 0
    else:
        print(f'Error: Person "{name}" not found in database', file=sys.stderr)
        return 1


def cmd_info(args: argparse.Namespace, config: AppConfig) -> int:
    """Show details about a registered person.

    Args:
        args: Parsed arguments with name.
        config: Application configuration.

    Returns:
        Exit code (0 for success, 1 if not found).
    """
    name: str = args.name.strip()

    if not name:
        print("Error: Name cannot be empty", file=sys.stderr)
        return 1

    database = JsonDatabase(config.database_path)
    person = database.get(name)

    if person is None:
        print(f'Error: Person "{name}" not found in database', file=sys.stderr)
        return 1

    print(f"Name: {person.name}")
    print(f"ID: {person.id}")
    print(f"Registered: {person.created_at}")

    # Show first few embedding values
    embedding_preview = ", ".join(f"{v:.4f}" for v in person.embedding[:5])
    print(f"Embedding: [{embedding_preview}, ...] ({len(person.embedding)} dimensions)")

    return 0
