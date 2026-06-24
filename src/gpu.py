"""Make a local conda-provided CUDA 11.2 / cuDNN 8.1 toolchain visible to TF 2.10.

TF 2.10 (the last native-Windows-GPU build) does not auto-discover CUDA DLLs, so
we add the conda env's bin dir to the DLL search path *before* TensorFlow is
imported. No-op if the CUDA env isn't present (CPU-only machines just skip it).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def enable_cuda_dlls():
    """Add the CUDA DLL directory to the search path if found. Returns the path or None."""
    for cand in (os.environ.get("CUDA_DLL_DIR"), ROOT / ".cuda" / "Library" / "bin"):
        if cand and Path(cand).is_dir():
            cand = str(cand)
            os.add_dll_directory(cand)
            os.environ["PATH"] = cand + os.pathsep + os.environ.get("PATH", "")
            return cand
    return None
