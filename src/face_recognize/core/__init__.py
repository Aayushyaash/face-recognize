"""Core face detection and tracking modules."""

import sys
from pathlib import Path

from .detector import FaceDetector
from .models import BoundingBox, Face
from .tracker import FaceTracker, Track
from .utils import compute_pairwise_iou, cosine_similarity, find_best_matches


# --- NVIDIA GPU SUPPORT FIX ---
# Must register DLLs *before* importing modules that use onnxruntime/cv2.
# On Windows, os.add_dll_directory() does NOT propagate to transitive C++
# LoadLibrary calls. ONNX Runtime's native code loads cublasLt64_12.dll etc.
# via the system PATH, so we must inject nvidia pip-package bin dirs there.
def _register_nvidia_dlls() -> None:
    """Register NVIDIA DLL paths for onnxruntime-gpu on Windows.

    Stage 1: Use onnxruntime.preload_dlls() (available since 1.21.0).
    Stage 2: Fallback to manual PATH injection for older ort versions.
    """
    if sys.platform != "win32":
        return

    # Stage 1: Try the official ONNX Runtime DLL preloader
    try:
        import onnxruntime

        if hasattr(onnxruntime, "preload_dlls"):
            # directory="" tells it to search NVIDIA site-packages
            onnxruntime.preload_dlls(directory="")
            # Don't return — still run manual PATH injection below
            # because preload_dlls may not cover all sub-packages (e.g. cuda_nvrtc)
    except ImportError:
        pass

    # Stage 2: Manual PATH injection (fallback for ort < 1.21)
    import os
    import site

    site_packages = site.getsitepackages() if hasattr(site, "getsitepackages") else []
    site_packages.append(os.path.join(sys.prefix, "Lib", "site-packages"))
    site_packages.append(os.path.join(sys.prefix, "site-packages"))

    nvidia_libs = [
        "nvidia/cublas/bin",
        "nvidia/cudnn/bin",
        "nvidia/cuda_runtime/bin",
        "nvidia/cuda_nvrtc/bin",
        "nvidia/cufft/bin",
        "nvidia/curand/bin",
        "nvidia/nvjitlink/bin",
    ]

    for site_pkg in site_packages:
        base_path = Path(site_pkg)
        for lib_rel_path in nvidia_libs:
            lib_path = base_path / lib_rel_path
            if lib_path.exists():
                try:
                    os.add_dll_directory(str(lib_path))
                except Exception:  # nosec B110
                    pass
                old_path = os.environ.get("PATH", "")
                os.environ["PATH"] = f"{lib_path}{os.pathsep}{old_path}"


_register_nvidia_dlls()
# ------------------------------

__all__ = [
    "BoundingBox",
    "Face",
    "FaceDetector",
    "FaceTracker",
    "Track",
    "compute_pairwise_iou",
    "cosine_similarity",
    "find_best_matches",
]
