"""PyTorch input pipeline with CXR-safe augmentation and TorchXRayVision preprocessing.

TorchXRayVision models expect a single-channel image normalized to the
[-1024, 1024] range (their ``xrv.datasets.normalize(img, 255)`` maps [0, 255] to
[-1024, 1024], i.e. ``pixel/255 * 2048 - 1024``). We reproduce that exactly with
a final ``t * 2048 - 1024`` step after ``ToTensor``.

Augmentation is intentionally anatomy-preserving: only small rotations, shifts,
zoom and mild intensity jitter. We deliberately do **not** use flips or 90°
rotations — a mirrored chest film moves the heart to the wrong side, which is
exactly wrong for a class like Cardiomegaly.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from . import config


def _to_xrv_range(t):
    """Map a [0, 1] tensor to TorchXRayVision's [-1024, 1024] range."""
    return t * 2048.0 - 1024.0


def _transforms(train: bool):
    steps = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(config.IMG_SIZE),
        transforms.CenterCrop(config.IMG_SIZE),
    ]
    if train:
        steps += [
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    steps += [transforms.ToTensor(), transforms.Lambda(_to_xrv_range)]
    return transforms.Compose(steps)


def _loader(directory, train):
    dataset = ImageFolder(str(directory), transform=_transforms(train))
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=train,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def build_loaders():
    """Return (train_loader, val_loader, test_loader, class_names)."""
    train_loader = _loader(config.TRAIN_DIR, train=True)
    val_loader = _loader(config.VAL_DIR, train=False)
    test_loader = _loader(config.TEST_DIR, train=False)
    class_names = train_loader.dataset.classes   # alphabetical: matches ImageFolder indices
    return train_loader, val_loader, test_loader, class_names


def class_weights(class_names):
    """Inverse-frequency weights (× optional manual boost) as a float tensor."""
    counts = []
    for name in class_names:
        class_dir = config.TRAIN_DIR / name
        n = sum(1 for p in class_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in config.IMAGE_SUFFIXES)
        counts.append(max(n, 1))
    total = sum(counts)
    n_classes = len(counts)
    weights = [total / (n_classes * c) for c in counts]
    for i, name in enumerate(class_names):
        weights[i] *= config.CLASS_WEIGHT_BOOST.get(name, 1.0)
    return torch.tensor(weights, dtype=torch.float32)
