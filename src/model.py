"""DenseNet121 backbone (CheXNet) + a small classifier head."""
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import DenseNet121

from . import config


def build_model(num_classes: int):
    """Return (model, base); ``base`` is exposed so it can be unfrozen for fine-tuning."""
    base = DenseNet121(weights="imagenet", include_top=False, input_shape=(*config.IMG_SIZE, 3))
    base.trainable = False
    inputs = layers.Input((*config.IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="densenet121_chestxray"), base


def unfreeze(base, at=config.FINE_TUNE_AT):
    """Unfreeze the backbone from layer ``at`` onward."""
    base.trainable = True
    for layer in base.layers[:at]:
        layer.trainable = False
