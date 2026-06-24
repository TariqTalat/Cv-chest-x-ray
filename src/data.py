"""tf.data input pipeline: DenseNet preprocessing + light augmentation.

Uses DenseNet's own ``preprocess_input`` (not ``rescale=1./255``), which is
what the ImageNet-pretrained backbone expects.
"""
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input

from . import config

AUTOTUNE = tf.data.AUTOTUNE

_AUGMENT = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", seed=config.SEED),
    tf.keras.layers.RandomRotation(0.05, seed=config.SEED),
    tf.keras.layers.RandomZoom(0.1, seed=config.SEED),
])


def _load(directory, shuffle):
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {directory}")
    return tf.keras.utils.image_dataset_from_directory(
        directory, label_mode="categorical", class_names=config.CLASS_NAMES,
        image_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE, shuffle=shuffle, seed=config.SEED)


def build_datasets():
    """Return (train, val, test) datasets ready for ``model.fit``."""
    prep = lambda ds: ds.map(lambda x, y: (preprocess_input(x), y), AUTOTUNE).prefetch(AUTOTUNE)
    train = _load(config.TRAIN_DIR, True).map(lambda x, y: (_AUGMENT(x, training=True), y), AUTOTUNE)
    return prep(train), prep(_load(config.VAL_DIR, False)), prep(_load(config.TEST_DIR, False))
