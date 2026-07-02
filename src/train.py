"""Two-phase training with a single LR schedule per phase, class weights, and AUC."""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks as cb
from tqdm.keras import TqdmCallback

from . import config
from .model import unfreeze


def _cosine_decay(epoch, total_epochs, initial_lr):
    """Cosine annealing over ``total_epochs`` (matched to the actual fine-tune length)."""
    return float(initial_lr) * 0.5 * (1.0 + math.cos(math.pi * epoch / max(total_epochs, 1)))


def _callbacks(phase="frozen"):
    """One learning-rate controller per phase — never a scheduler *and* a plateau reducer."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cbs = [
        cb.ModelCheckpoint(str(config.BEST_MODEL_PATH), monitor="val_loss",
                           mode="min", save_best_only=True, verbose=0),
        cb.EarlyStopping(monitor="val_loss", patience=8,
                         restore_best_weights=True, verbose=1),
        TqdmCallback(verbose=1),
    ]
    if phase == "finetune":
        cbs.insert(0, cb.LearningRateScheduler(
            lambda epoch, lr: _cosine_decay(epoch, config.EPOCHS_FINETUNE, config.LR_FINETUNE),
            verbose=0))
    else:
        cbs.insert(0, cb.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                           patience=3, min_lr=1e-7, verbose=1))
    return cbs


def _compile(model, lr):
    """Compile with the metrics that actually make sense for multiclass CXR.

    We report accuracy and (macro) AUC. Per-class precision/recall come from the
    classification report in ``evaluate.py`` — the built-in Precision/Recall
    metrics threshold a softmax at 0.5 and are meaningless for >2 classes.
    """
    metrics = ["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)]
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=metrics)


def train(model, base, train_ds, val_ds, finetune=True, class_weight=None):
    """Phase 1: train the head (backbone frozen). Phase 2: fine-tune the backbone."""
    print("\n" + "=" * 70)
    print("=== Phase 1: Training classifier head (backbone frozen) ===")
    print("=" * 70)
    _compile(model, config.LR_FROZEN)
    hist = [model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS_FROZEN,
        callbacks=_callbacks("frozen"),
        class_weight=class_weight,
        verbose=0,
    )]

    if finetune:
        print("\n" + "=" * 70)
        print("=== Phase 2: Fine-tuning the backbone's last dense block ===")
        print("=" * 70)
        unfreeze(base, at=config.FINE_TUNE_AT)
        _compile(model, config.LR_FINETUNE)
        hist.append(model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.EPOCHS_FINETUNE,
            callbacks=_callbacks("finetune"),
            class_weight=class_weight,
            verbose=0,
        ))

    return hist


def plot_history(histories):
    """Save training curves for the metrics we track."""
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pick = lambda key: [v for h in histories for v in h.history.get(key, [])]

    metrics = [("accuracy", "Accuracy"), ("loss", "Loss"), ("auc", "AUC")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (metric, title) in zip(axes.flat, metrics):
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
