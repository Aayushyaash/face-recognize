"""Main entry point for the face-recognize CLI.

This module defines the argument parser and dispatches to command handlers.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from ..config import AppConfig
from .commands import cmd_delete, cmd_info, cmd_list, cmd_register, cmd_run


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="face-recognize",
        description="Real-time face identification system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  face-recognize register photo.jpg "John Smith"
  face-recognize list
  face-recognize run --camera 0
  face-recognize info "John Smith"
  face-recognize delete "John Smith"
        """,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="commands",
        description="Available commands",
    )

    # === run command ===
    run_parser = subparsers.add_parser(
        "run",
        help="Start real-time camera identification",
        description="Start the camera and identify faces in real-time.",
    )
    run_parser.add_argument(
        "--camera",
        type=str,
        default="0",
        metavar="SOURCE",
        help="Camera index (0, 1...) or network URL (http://...)",
    )
    run_parser.add_argument(
        "--model",
        type=str,
        default="buffalo_s",
        choices=["buffalo_s", "buffalo_l", "buffalo_sc"],
        help="InsightFace model to use (default: buffalo_s)",
    )
    run_parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        metavar="SCORE",
        help="Similarity threshold for identification (default: 0.4)",
    )
    run_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device (default: cpu)",
    )

    # === register command ===
    register_parser = subparsers.add_parser(
        "register",
        help="Register a new person from an image",
        description="Detect a face in the image and register it with the given name.",
    )
    register_parser.add_argument(
        "image_path",
        type=Path,
        metavar="IMAGE",
        help="Path to the image file containing the face",
    )
    register_parser.add_argument(
        "name",
        type=str,
        metavar="NAME",
        help="Name to associate with the face",
    )

    # === list command ===
    subparsers.add_parser(
        "list",
        help="List all registered persons",
        description="Display all persons registered in the database.",
    )

    # === delete command ===
    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete a person from the database",
        description="Remove a person and their face embedding from the database.",
    )
    delete_parser.add_argument(
        "name",
        type=str,
        metavar="NAME",
        help="Name of the person to delete",
    )

    # === info command ===
    info_parser = subparsers.add_parser(
        "info",
        help="Show details about a registered person",
        description="Display detailed information about a registered person.",
    )
    info_parser.add_argument(
        "name",
        type=str,
        metavar="NAME",
        help="Name of the person to look up",
    )

    return parser


def main() -> int:
    """Main entry point for CLI.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = create_parser()
    args = parser.parse_args()

    # Create config with any command-line overrides
    config = AppConfig()

    # Apply command-line overrides if running the 'run' command
    if args.command == "run":
        config = replace(
            config,
            camera_index=args.camera,
            model=args.model,
            similarity_threshold=args.threshold,
            device=args.device,
        )

    # Initialize logging
    from ..core.logger import setup_logger

    log_file = Path("logs/face_recognize.log")
    setup_logger(log_file=log_file)

    # Dispatch to appropriate command handler
    try:
        if args.command == "run":
            return cmd_run(args, config)
        elif args.command == "register":
            return cmd_register(args, config)
        elif args.command == "list":
            return cmd_list(args, config)
        elif args.command == "delete":
            return cmd_delete(args, config)
        elif args.command == "info":
            return cmd_info(args, config)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
