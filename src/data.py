"""tf.data input pipeline with CXR-safe augmentation and DenseNet preprocessing.

Augmentation is intentionally anatomy-preserving: only small rotations, zoom,
shifts and mild intensity jitter. We deliberately do **not** use 90° rotations
or left/right flips — a sideways/mirrored chest film never occurs in reality,
and a horizontal flip mislocates the heart, which is exactly wrong for a class
like Cardiomegaly.
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.densenet import preprocess_input

from . import config

AUTOTUNE = tf.data.AUTOTUNE

# Keras preprocessing layers handle value ranges correctly (unlike hand-rolled
# ops on a 0-255 image, where a ±0.1 brightness delta is invisible).
_augment = tf.keras.Sequential(
    [
        layers.RandomRotation(0.03, fill_mode="constant"),        # ~±10 degrees
        layers.RandomZoom(0.1, fill_mode="constant"),
        layers.RandomTranslation(0.05, 0.05, fill_mode="constant"),
        layers.RandomContrast(0.1),
    ],
    name="cxr_augment",
)


def _load(directory, shuffle):
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {directory}")
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        label_mode="categorical",
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        seed=config.SEED,
    )


def build_datasets():
    """Return (train, val, test, class_names) datasets."""
    # Quick sanity check: ensure the prepared train dir contains images
    def _count_images_in_dir(directory):
        counts = {}
        for path in sorted(directory.iterdir()):
            if not path.is_dir() or path.name.startswith("_"):
                continue
            n = sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in config.IMAGE_SUFFIXES)
            counts[path.name] = n
        return counts

    train_counts = _count_images_in_dir(config.TRAIN_DIR)
    total_train = sum(train_counts.values())
    if total_train == 0:
        raise FileNotFoundError(
            f"No images found in prepared train folder: {config.TRAIN_DIR}\n"
            f"Check your CXR_DATA_ROOT or provide a pre-split dataset. Detected class folders: {list(train_counts.items())}"
        )

    train = _load(config.TRAIN_DIR, True)
    val = _load(config.VAL_DIR, False)
    test = _load(config.TEST_DIR, False)
    class_names = train.class_names

    def preprocess_fn(x, y):
        return preprocess_input(x), y

    def augment_fn(x, y):
        x = tf.cast(x, tf.float32)
        x = _augment(x, training=True)
        x = tf.image.random_brightness(x, max_delta=15.0)
        x = tf.clip_by_value(x, 0.0, 255.0)
        return preprocess_input(x), y

    train = train.map(augment_fn, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val = val.map(preprocess_fn, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test = test.map(preprocess_fn, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train, val, test, class_names


def compute_class_weights(class_names):
    """Inverse-frequency class weights from the training folder (handles imbalance)."""
    counts = []
    for name in class_names:
        class_dir = config.TRAIN_DIR / name
        n = sum(
            1
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in config.IMAGE_SUFFIXES
        )
        counts.append(max(n, 1))
    total = sum(counts)
    n_classes = len(counts)
    return {i: total / (n_classes * c) for i, c in enumerate(counts)}
