"""Model, two-phase training, and evaluation for the chest X-ray classifier.

Architecture is the proven CheXNet-style baseline: a DenseNet121 backbone
(ImageNet) with a light global-pool head. The lever that actually matters on a
small dataset is the *training recipe*, not head complexity:

  Phase 1  warm up the head with the backbone frozen (high LR).
  Phase 2  fine-tune the last dense blocks with a low, cosine-decayed LR.

Backbone BatchNorm always runs in inference mode (``training=False``), so its
ImageNet running statistics stay frozen even while conv weights are fine-tuned —
the recommended setup for transfer learning on limited data.
"""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import Model, layers
from tensorflow.keras import callbacks as cb
from tensorflow.keras.applications import DenseNet121
from tqdm.keras import TqdmCallback

from . import config


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(num_classes: int):
    """Return ``(model, base)``; ``base`` is exposed so it can be unfrozen later."""
    base = DenseNet121(weights="imagenet", include_top=False, input_shape=(*config.IMG_SIZE, 3))
    base.trainable = False

    inputs = layers.Input((*config.IMG_SIZE, 3))
    x = base(inputs, training=False)          # keep backbone BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="densenet121_cxr"), base


def unfreeze(base, from_block=config.FINE_TUNE_FROM):
    """Unfreeze the backbone from the given dense block onward (e.g. ``"conv4"``)."""
    base.trainable = True
    trainable = False
    for layer in base.layers:
        if layer.name.startswith(from_block):
            trainable = True
        layer.trainable = trainable


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def categorical_focal_loss(gamma):
    """Softmax focal loss — down-weights easy examples (1 - p)^gamma so training
    focuses on the hard, misclassified cases instead of coasting on easy ones."""
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        return tf.reduce_sum(tf.pow(1.0 - y_pred, gamma) * ce, axis=-1)
    return loss


def _compile(model, lr):
    """Accuracy + macro AUC. Per-class precision/recall come from evaluate()."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=categorical_focal_loss(config.FOCAL_GAMMA),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )


def _callbacks(phase):
    """One learning-rate controller per phase — never a scheduler *and* a plateau reducer."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Monitor val AUC (not val_loss): threshold-free and robust to label
    # smoothing, so fine-tuning isn't cut short by a noisy loss curve.
    cbs = [
        cb.ModelCheckpoint(str(config.BEST_MODEL_PATH), monitor="val_auc",
                           mode="max", save_best_only=True, verbose=0),
        cb.EarlyStopping(monitor="val_auc", mode="max", patience=10,
                         restore_best_weights=True, verbose=1),
        TqdmCallback(verbose=1),
    ]
    if phase == "finetune":
        cbs.insert(0, cb.LearningRateScheduler(
            lambda epoch, lr: config.LR_FINETUNE * 0.5
            * (1.0 + math.cos(math.pi * epoch / max(config.EPOCHS_FINETUNE, 1)))))
    else:
        cbs.insert(0, cb.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                           patience=2, min_lr=1e-7, verbose=1))
    return cbs


def _fit(model, train_ds, val_ds, epochs, phase, class_weight):
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     callbacks=_callbacks(phase), class_weight=class_weight, verbose=0)


def train(model, base, train_ds, val_ds, finetune=True, class_weight=None):
    """Phase 1: train the head (backbone frozen). Phase 2: fine-tune the backbone."""
    print("\n" + "=" * 70 + "\n=== Phase 1: Training classifier head (backbone frozen) ===\n" + "=" * 70)
    _compile(model, config.LR_FROZEN)
    histories = [_fit(model, train_ds, val_ds, config.EPOCHS_FROZEN, "frozen", class_weight)]

    if finetune:
        print("\n" + "=" * 70 + f"\n=== Phase 2: Fine-tuning the backbone (from {config.FINE_TUNE_FROM}) ===\n" + "=" * 70)
        unfreeze(base)
        _compile(model, config.LR_FINETUNE)
        histories.append(_fit(model, train_ds, val_ds, config.EPOCHS_FINETUNE, "finetune", class_weight))

    return histories


def plot_history(histories):
    """Save train/val curves for accuracy, loss and AUC."""
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pick = lambda key: [v for h in histories for v in h.history.get(key, [])]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (metric, title) in zip(axes.flat, [("accuracy", "Accuracy"), ("loss", "Loss"), ("auc", "AUC")]):
        train_vals = pick(metric)
        if not train_vals:
            continue
        epochs = range(1, len(train_vals) + 1)
        ax.plot(epochs, train_vals, label="train", linewidth=2)
        val_vals = pick("val_" + metric)
        if val_vals:
            ax.plot(epochs, val_vals, label="val", linewidth=2)
        ax.set(title=title, xlabel="Epoch", ylabel=title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = config.OUTPUTS_DIR / "training_history.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Saved training curves → {out}")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(model, test_ds, class_names):
    """Evaluate on the test set; write report + confusion matrix to outputs/."""
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    y_true = np.concatenate([y.numpy() for _, y in test_ds]).argmax(1)
    y_pred = model.predict(test_ds, verbose=0).argmax(1)
    acc = float((y_true == y_pred).mean())
    print(f"\nTest accuracy: {acc:.4f}")

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n" + report)
    (config.OUTPUTS_DIR / "classification_report.txt").write_text(
        f"Test accuracy: {acc:.4f}\n\n{report}", encoding="utf-8")

    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix"); plt.tight_layout()
    plt.savefig(config.OUTPUTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved report + confusion matrix -> {config.OUTPUTS_DIR}")
    return acc
