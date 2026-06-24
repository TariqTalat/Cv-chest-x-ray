"""Two-phase training (frozen head -> fine-tune) and the training-curve plot."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks as cb
from tqdm.keras import TqdmCallback

from . import config
from .model import unfreeze


def _callbacks():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return [
        cb.ModelCheckpoint(str(config.BEST_MODEL_PATH), monitor="val_accuracy",
                           mode="max", save_best_only=True, verbose=0),
        cb.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        cb.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1),
        TqdmCallback(verbose=1),   # tqdm progress bars (epoch + per-batch)
    ]


def _compile(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])


def train(model, base, train_ds, val_ds, finetune=True):
    """Phase 1 (frozen) then optional phase 2 (fine-tune). Returns list of Histories."""
    callbacks = _callbacks()
    _compile(model, config.LR_FROZEN)
    print("\n=== Phase 1: training head (backbone frozen) ===")
    hist = [model.fit(train_ds, validation_data=val_ds, epochs=config.EPOCHS_FROZEN,
                      callbacks=callbacks, verbose=0)]
    if finetune:
        print("\n=== Phase 2: fine-tuning top of DenseNet121 ===")
        unfreeze(base)
        _compile(model, config.LR_FINETUNE)
        hist.append(model.fit(train_ds, validation_data=val_ds, epochs=config.EPOCHS_FINETUNE,
                              callbacks=callbacks, verbose=0))
    return hist


def plot_history(histories):
    """Save combined accuracy/loss curves across both phases."""
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pick = lambda key: [v for h in histories for v in h.history.get(key, [])]
    epochs = range(1, len(pick("loss")) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ("accuracy", "loss")):
        ax.plot(epochs, pick(metric), label="train")
        ax.plot(epochs, pick("val_" + metric), label="val")
        ax.set(title=metric.capitalize(), xlabel="epoch")
        ax.legend()
    fig.tight_layout()
    out = config.OUTPUTS_DIR / "training_history.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved curves -> {out}")
