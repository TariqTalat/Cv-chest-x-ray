"""Test-set evaluation: accuracy, classification report, confusion matrix."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from . import config


def evaluate(model, test_ds):
    """Evaluate on the (unshuffled) test set; write report + confusion matrix to outputs/."""
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}   loss: {loss:.4f}")

    y_true = np.concatenate([y.numpy() for _, y in test_ds]).argmax(1)
    y_pred = model.predict(test_ds, verbose=0).argmax(1)
    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, digits=4)
    print("\n" + report)
    (config.OUTPUTS_DIR / "classification_report.txt").write_text(
        f"Test accuracy: {acc:.4f}\nTest loss: {loss:.4f}\n\n{report}", encoding="utf-8")

    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix"); plt.tight_layout()
    plt.savefig(config.OUTPUTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved report + confusion matrix -> {config.OUTPUTS_DIR}")
    return acc
