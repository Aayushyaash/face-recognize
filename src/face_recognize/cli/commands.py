"""Command implementations for the face-recognize CLI.

Each command function takes parsed args and config, returns exit code.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from ..config import AppConfig
from ..core.camera import Camera
from ..core.detector import FaceDetector
from ..core.logger import logger
from ..core.quality import FaceQualityScorer
from ..core.threaded_camera import ThreadedCamera
from ..core.tracker import FaceTracker
from ..database import create_database
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
    # Parse camera sources (args.camera is now a list)
    camera_sources = args.camera
    # Convert string digits to integers where appropriate
    sources: list[int | str] = []
    for source in camera_sources:
        if source.isdigit():
            sources.append(int(source))
        else:
            sources.append(source)

    # Apply command-line overrides to config
    config = replace(
        config,
        camera_index=(
            sources[0] if len(sources) == 1 else sources
        ),  # Keep first camera or list
        model=args.model,
        similarity_threshold=args.threshold,
    )
    logger.info(f"Initializing with model: {config.model}")
    logger.info(f"Similarity threshold: {config.similarity_threshold}")
    logger.info(f"Using cameras: {sources}")

    # Initialize components
    try:
        detector = FaceDetector(config)
    except Exception as e:
        logger.error(f"Error initializing face detector: {e}")
        logger.error("Make sure the model is downloaded to ./models/")
        return 1

    tracker = FaceTracker(config)
    database = create_database(config)
    identifier = IdentificationService(database, config)
    renderer = FaceRenderer(config)
    quality_scorer = FaceQualityScorer()

    logger.info(f"Database loaded: {database.count()} persons")

    # FPS calculation
    frame_times: list[float] = []
    fps = 0.0

    try:
        # Handle single vs multiple cameras
        if len(sources) == 1:
            # Single camera mode - use original implementation
            source = sources[0]
            with Camera(source, config.frame_width, config.frame_height) as camera:
                logger.info("Press 'q' to quit.")

                while True:
                    frame_start = time.time()

                    # Capture frame
                    frame = camera.read()
                    if frame is None:
                        # Error already logged by Camera class
                        break

                    # Detect faces
                    faces = detector.detect_faces(frame)

                    # Evaluate quality for each face and update with quality scores
                    for face in faces:
                        # Cast frame to correct type for quality scorer
                        frame_uint8 = frame.astype(np.uint8)
                        quality_score = quality_scorer.evaluate(frame_uint8, face.bbox)
                        face.quality_score = quality_score

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
                        logger.info("Quitting...")
                        break
        else:
            # Multiple camera mode
            logger.info(f"Starting multi-camera mode with {len(sources)} cameras")

            # Create threaded cameras
            threaded_cameras = []
            for source in sources:
                cam = ThreadedCamera(
                    source=source,
                    frame_width=config.frame_width,
                    frame_height=config.frame_height,
                )
                threaded_cameras.append(cam)

            # Start all cameras
            for cam in threaded_cameras:
                cam.start()

            logger.info("Press 'q' to quit.")

            try:
                while True:
                    frame_start = time.time()

                    # Capture frames from all cameras
                    frames = []
                    for cam in threaded_cameras:
                        frame = cam.read()
                        if frame is not None:
                            frames.append(frame)

                    if not frames:
                        # No frames available, wait a bit and continue
                        time.sleep(0.01)
                        continue

                    # Process each frame for face detection
                    all_identified_faces = []
                    processed_frames = []

                    for i, frame in enumerate(frames):
                        # Detect faces
                        faces = detector.detect_faces(frame)

                        # Evaluate quality for each face and update with quality scores
                        for face in faces:
                            # Cast frame to correct type for quality scorer
                            frame_uint8 = frame.astype(np.uint8)
                            quality_score = quality_scorer.evaluate(
                                frame_uint8, face.bbox
                            )
                            face.quality_score = quality_score

                        # Track faces
                        tracked_faces = tracker.update(faces)

                        # Identify faces
                        identified_faces = identifier.identify(tracked_faces)

                        all_identified_faces.extend(identified_faces)

                        # Render on this frame
                        renderer.render(frame, identified_faces)
                        processed_frames.append(frame)

                    # Create grid view
                    grid_frame = renderer.render_grid_view(processed_frames)

                    # Add FPS to grid view
                    renderer.render_fps(grid_frame, fps)

                    # Display grid view
                    cv2.imshow("Face-Recognize - Multi-Camera Grid", grid_frame)

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
                        logger.info("Quitting...")
                        break

            finally:
                # Stop all cameras
                for cam in threaded_cameras:
                    cam.stop()

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    finally:
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
        logger.error(f"Error: Image file not found: {image_path}")
        return 1

    if not name:
        logger.error("Error: Name cannot be empty")
        return 1

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Error: Could not read image: {image_path}")
        return 1

    logger.info(f"Processing image: {image_path}")

    # Initialize detector
    try:
        detector = FaceDetector(config)
    except Exception as e:
        logger.error(f"Error initializing detector: {e}")
        return 1

    # Detect faces
    faces = detector.detect_faces(image)

    # Validate exactly one face
    if len(faces) == 0:
        logger.error("Error: No face detected in image")
        return 1

    if len(faces) > 1:
        logger.error(
            f"Error: Multiple faces detected ({len(faces)}). "
            "Please use an image with exactly one face."
        )
        return 1

    face = faces[0]

    # Evaluate face quality using the quality scorer
    quality_scorer = FaceQualityScorer()
    quality_score = quality_scorer.evaluate(image, face.bbox)

    # Validate face quality
    if quality_score < config.min_quality_score:
        logger.error(
            f"Error: Face quality too low ({quality_score:.2f}). "
            f"Minimum required: {config.min_quality_score}"
        )
        return 1

    # Add to database
    database = create_database(config)

    try:
        record = database.add(name, face.embedding)
        logger.info(f'✓ Registered "{name}" successfully (ID: {record.id[:8]})')
        return 0
    except ValueError as e:
        logger.error(f"Error: {e}")
        return 1


def cmd_list(args: argparse.Namespace, config: AppConfig) -> int:
    """List all registered persons.

    Args:
        args: Parsed arguments (unused).
        config: Application configuration.

    Returns:
        Exit code (0 for success).
    """
    database = create_database(config)
    persons = database.list_all()

    if not persons:
        logger.info("No registered persons.")
        return 0

    logger.info(f"Registered Persons ({len(persons)}):")
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

        logger.info(f"  {i}. {person.name:<20} (registered: {formatted_time})")

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
        logger.error("Error: Name cannot be empty")
        return 1

    database = create_database(config)

    if database.delete(name):
        logger.info(f'✓ Deleted "{name}" from database')
        return 0
    else:
        logger.error(f'Error: Person "{name}" not found in database')
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
        logger.error("Error: Name cannot be empty")
        return 1

    database = create_database(config)
    person = database.get(name)

    if person is None:
        logger.error(f'Error: Person "{name}" not found in database')
        return 1

    logger.info(f"Name: {person.name}")
    logger.info(f"ID: {person.id}")
    logger.info(f"Registered: {person.created_at}")

    # Show first few embedding values
    embedding_preview = ", ".join(f"{v:.4f}" for v in person.embedding[:5])
    logger.info(
        f"Embedding: [{embedding_preview}, ...] ({len(person.embedding)} dimensions)"
    )

    return 0
