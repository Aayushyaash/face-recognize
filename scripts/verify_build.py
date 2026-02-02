#!/usr/bin/env python3
"""
Build verification script.

This script verifies that the package can be properly built and installed from source.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def test_package_build_and_install() -> bool:
    """Test that the package can be built and installed."""
    try:
        # Build the package
        print("Building the package...")
        result = subprocess.run(
            [sys.executable, "-m", "build"], capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            return False

        print("Package built successfully!")

        # Check if dist directory exists and has files
        dist_dir = Path("dist")
        if not dist_dir.exists():
            print("Build completed but no dist directory found")
            return False

        dist_files = list(dist_dir.glob("*"))
        if not dist_files:
            print("Build completed but no distribution files found")
            return False

        print(f"Found distribution files: {[f.name for f in dist_files]}")

        # Create a temporary directory for testing installation
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a virtual environment in the temp directory
            print("Creating temporary virtual environment...")
            venv_result = subprocess.run(
                [sys.executable, "-m", "venv", str(temp_path / "test_env")],
                capture_output=True,
                text=True,
            )

            if venv_result.returncode != 0:
                print(f"Failed to create virtual environment: {venv_result.stderr}")
                return False

            # Determine the path to pip in the virtual environment
            if os.name == "nt":  # Windows
                pip_path = temp_path / "test_env" / "Scripts" / "pip"
            else:  # Unix/Linux/macOS
                pip_path = temp_path / "test_env" / "bin" / "pip"

            # Install the built package in the virtual environment
            print("Installing package in virtual environment...")
            wheel_files = list(dist_dir.glob("*.whl"))
            tar_files = list(dist_dir.glob("*.tar.gz"))

            all_dist_files = wheel_files + tar_files
            if not all_dist_files:
                print("No distribution files found to install")
                return False

            install_targets = [str(f) for f in all_dist_files]

            install_result = subprocess.run(
                [str(pip_path), "install"] + install_targets,
                capture_output=True,
                text=True,
            )

            if install_result.returncode != 0:
                print(f"Installation failed: {install_result.stderr}")
                return False

            print("Package installed successfully in virtual environment!")

            # Test importing the package
            if os.name == "nt":  # Windows
                python_path = temp_path / "test_env" / "Scripts" / "python"
            else:  # Unix/Linux/macOS
                python_path = temp_path / "test_env" / "bin" / "python"

            import_test = subprocess.run(
                [
                    str(python_path),
                    "-c",
                    "import face_recognize; print('Import successful')",
                ],
                capture_output=True,
                text=True,
            )

            if import_test.returncode != 0:
                print(f"Import test failed: {import_test.stderr}")
                return False

            print("Package import test passed!")

        return True
    except Exception as e:
        print(f"Error during build verification: {str(e)}")
        return False


def main() -> int:
    """Main function to run build verification."""
    print("Verifying package build and installation...")

    success = test_package_build_and_install()

    if success:
        print("Build verification passed!")
        return 0
    else:
        print("Build verification failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
