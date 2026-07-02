"""DenseNet121 (CheXNet-style) transfer-learning classifier.

Kept deliberately light: a strong pretrained backbone plus a single global-pool
head is the proven baseline for chest X-ray classification (this is essentially
the CheXNet design). A small dataset does not support a heavy attention/dense
head without overfitting, so we don't add one.
"""
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import DenseNet121

from . import config


def build_model(num_classes: int):
    """Return ``(model, base)``; ``base`` is exposed so it can be unfrozen later."""
    base = DenseNet121(weights="imagenet", include_top=False,
                       input_shape=(*config.IMG_SIZE, 3))
    base.trainable = False

    inputs = layers.Input((*config.IMG_SIZE, 3))
    x = base(inputs, training=False)          # keep backbone BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="densenet121_cxr")
    return model, base


def unfreeze(base, at=config.FINE_TUNE_AT):
    """Unfreeze the backbone from layer ``at`` onward for fine-tuning.

    Backbone BatchNorm still runs in inference mode (see ``training=False`` in
    ``build_model``), so its ImageNet running statistics stay frozen — the
    recommended setup when fine-tuning a pretrained model on a small dataset.
    """
    base.trainable = True
    for layer in base.layers[:at]:
        layer.trainable = False
