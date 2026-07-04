"""Model, two-phase training, and evaluation for the chest X-ray classifier.

Backbone: a DenseNet121 pretrained on 100k+ chest X-rays via TorchXRayVision —
its features already encode findings like Nodule/Mass/Cardiomegaly, which is the
whole point of switching away from ImageNet. We attach a light 3-class head:

  Phase 1  freeze the backbone, warm up the head.
  Phase 2  fine-tune the last two dense blocks with a low, cosine-decayed LR.

The backbone runs in eval mode throughout (``features.eval()``) so its
pretrained BatchNorm statistics stay fixed — the standard transfer-learning
setup on a small dataset.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm

from . import config


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class CXRClassifier(nn.Module):
    """TorchXRayVision DenseNet121 features + global-pool + linear head."""

    def __init__(self, num_classes: int):
        super().__init__()
        backbone = xrv.models.DenseNet(weights=config.XRV_WEIGHTS)
        self.features = backbone.features
        feat_dim = getattr(backbone.classifier, "in_features", 1024)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        f = F.relu(self.features(x), inplace=True)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.classifier(self.dropout(f))


def _freeze_backbone(model):
    for p in model.features.parameters():
        p.requires_grad = False


def _unfreeze_finetune(model):
    """Unfreeze only the last dense blocks (config.FINETUNE_PREFIXES)."""
    for name, p in model.features.named_parameters():
        p.requires_grad = name.startswith(config.FINETUNE_PREFIXES)


def _set_mode(model, training: bool):
    model.train(training)      # toggles head dropout
    model.features.eval()      # always use the pretrained BatchNorm statistics


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def _make_criterion(weight):
    """Weighted cross-entropy, or focal loss when config.FOCAL_GAMMA > 0."""
    if config.FOCAL_GAMMA > 0:
        gamma = config.FOCAL_GAMMA

        def focal(logits, y):
            logp = F.log_softmax(logits, dim=1)
            ce = F.nll_loss(logp, y, weight=weight, reduction="none")
            pt = logp.gather(1, y[:, None]).squeeze(1).exp()
            return ((1.0 - pt) ** gamma * ce).mean()

        return focal
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=config.LABEL_SMOOTHING)


# --------------------------------------------------------------------------- #
# Training / evaluation loop
# --------------------------------------------------------------------------- #
def _run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    _set_mode(model, training)
    total_loss, probs, targets = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        probs.append(torch.softmax(logits, dim=1).detach().cpu())
        targets.append(y.cpu())
    probs = torch.cat(probs).numpy()
    targets = torch.cat(targets).numpy()
    return total_loss / len(targets), probs, targets


def _macro_auc(targets, probs, n_classes):
    try:
        return roc_auc_score(targets, probs, multi_class="ovr",
                             average="macro", labels=list(range(n_classes)))
    except ValueError:
        return float("nan")


def _fit_phase(model, phase, epochs, lr, train_loader, val_loader, device,
               criterion, n_classes, history):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                 if phase == "finetune" else None)

    best_auc, best_state, since_improved = -1.0, None, 0
    bar = tqdm(range(1, epochs + 1), desc=f"{phase}", unit="epoch")
    for _ in bar:
        tr_loss, tr_probs, tr_targets = _run_epoch(model, train_loader, device, criterion, optimizer)
        va_loss, va_probs, va_targets = _run_epoch(model, val_loader, device, criterion)
        if scheduler:
            scheduler.step()

        tr_acc = (tr_probs.argmax(1) == tr_targets).mean()
        va_acc = (va_probs.argmax(1) == va_targets).mean()
        va_auc = _macro_auc(va_targets, va_probs, n_classes)
        history["acc"].append(tr_acc); history["val_acc"].append(va_acc)
        history["loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["auc"].append(_macro_auc(tr_targets, tr_probs, n_classes))
        history["val_auc"].append(va_auc)
        bar.set_postfix(loss=f"{tr_loss:.3f}", acc=f"{tr_acc:.3f}",
                        val_acc=f"{va_acc:.3f}", val_auc=f"{va_auc:.3f}")

        if va_auc > best_auc:
            best_auc, best_state, since_improved = va_auc, _cpu_state(model), 0
        else:
            since_improved += 1
            if since_improved >= config.EARLY_STOP_PATIENCE:
                print(f"Early stopping ({phase}) — best val AUC {best_auc:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, config.BEST_MODEL_PATH)


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train(model, train_loader, val_loader, device, weight, n_classes):
    """Phase 1: train the head. Phase 2: fine-tune the last dense blocks."""
    criterion = _make_criterion(weight.to(device))
    history = {k: [] for k in ("acc", "val_acc", "loss", "val_loss", "auc", "val_auc")}

    print("\n" + "=" * 70 + "\n=== Phase 1: Training classifier head (backbone frozen) ===\n" + "=" * 70)
    _freeze_backbone(model)
    _fit_phase(model, "frozen", config.EPOCHS_FROZEN, config.LR_FROZEN,
               train_loader, val_loader, device, criterion, n_classes, history)

    if config.EPOCHS_FINETUNE > 0:
        print("\n" + "=" * 70 + "\n=== Phase 2: Fine-tuning the last dense blocks ===\n" + "=" * 70)
        _unfreeze_finetune(model)
        _fit_phase(model, "finetune", config.EPOCHS_FINETUNE, config.LR_FINETUNE,
                   train_loader, val_loader, device, criterion, n_classes, history)

    return history


def plot_history(history):
    """Save train/val curves for accuracy, loss and AUC."""
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (metric, title) in zip(axes.flat, [("acc", "Accuracy"), ("loss", "Loss"), ("auc", "AUC")]):
        train_vals = history[metric]
        if not train_vals:
            continue
        epochs = range(1, len(train_vals) + 1)
        ax.plot(epochs, train_vals, label="train", linewidth=2)
        ax.plot(epochs, history["val_" + metric], label="val", linewidth=2)
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
@torch.no_grad()
def evaluate(model, test_loader, class_names, device):
    """Evaluate on the test set; write report + confusion matrix to outputs/."""
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _set_mode(model, False)
    probs, targets = [], []
    for x, y in test_loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
        targets.append(y)
    y_prob = torch.cat(probs).numpy()
    y_true = torch.cat(targets).numpy()
    y_pred = y_prob.argmax(1)

    acc = float((y_pred == y_true).mean())
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
