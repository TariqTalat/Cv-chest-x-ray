"""Run the chest X-ray pipeline: train (2 phases) + evaluate.

All settings live in src/config.py. Just run:  uv run python main.py
"""
from src.gpu import enable_cuda_dlls

enable_cuda_dlls()  # must run before TensorFlow is imported (no-op without a local CUDA env)

import tensorflow as tf

from src import config, data, evaluate
from src import model as model_mod
from src import train as train_mod


def main():
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)  # don't grab all VRAM on a small GPU
    print(f"TensorFlow {tf.__version__} | GPU: {bool(tf.config.list_physical_devices('GPU'))}")

    train_ds, val_ds, test_ds, class_names = data.build_datasets()
    print("Classes:", class_names)

    class_weight = data.compute_class_weights(class_names)
    print("Class weights:", class_weight)

    model, base = model_mod.build_model(len(class_names))
    histories = train_mod.train(
        model, base, train_ds, val_ds,
        finetune=config.EPOCHS_FINETUNE > 0,
        class_weight=class_weight,
    )
    train_mod.plot_history(histories)

    if config.BEST_MODEL_PATH.exists():
        model = tf.keras.models.load_model(config.BEST_MODEL_PATH)
    evaluate.evaluate(model, test_ds, class_names)


if __name__ == "__main__":
    main()