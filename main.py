"""Run the chest X-ray pipeline: train (2 phases) + evaluate.

All settings live in src/config.py. Just run:  uv run python main.py
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _enable_cuda_dlls():
    """Make a local conda CUDA 11.2 / cuDNN 8.1 toolchain visible to TF 2.10 on Windows.

    TF 2.10 (last native-Windows-GPU build) doesn't auto-discover CUDA DLLs, so we
    add them to the search path *before* importing TensorFlow. No-op elsewhere.
    """
    for cand in (os.environ.get("CUDA_DLL_DIR"), ROOT / ".cuda" / "Library" / "bin"):
        if cand and Path(cand).is_dir() and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(cand))
            os.environ["PATH"] = f"{cand}{os.pathsep}{os.environ.get('PATH', '')}"
            return


_enable_cuda_dlls()  # must run before TensorFlow is imported

import tensorflow as tf

from src import config, data, pipeline


def main():
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)  # don't grab all VRAM on a small GPU
    print(f"TensorFlow {tf.__version__} | GPU: {bool(tf.config.list_physical_devices('GPU'))}")

    train_ds, val_ds, test_ds, class_names = data.build_datasets()
    print("Classes:", class_names)

    class_weight = data.compute_class_weights(class_names)
    print("Class weights:", class_weight)

    model, base = pipeline.build_model(len(class_names))
    histories = pipeline.train(
        model, base, train_ds, val_ds,
        finetune=config.EPOCHS_FINETUNE > 0,
        class_weight=class_weight,
    )
    pipeline.plot_history(histories)

    if config.BEST_MODEL_PATH.exists():
        model = tf.keras.models.load_model(config.BEST_MODEL_PATH, compile=False)
    pipeline.evaluate(model, test_ds, class_names)


if __name__ == "__main__":
    main()
