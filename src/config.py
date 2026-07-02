"""Paths, classes and hyper-parameters — the single place to tweak a run.

Point the pipeline at your data with the ``CXR_DATA_ROOT`` env var. It may be
either a folder already split into ``train/``, ``val/`` and ``test/``, or a
single folder with one sub-folder per class (a stratified 80/10/10 split is then
created once under ``_prepared_splits/``).
"""
import os
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _split_class_dirs(source_root: Path) -> tuple[Path, Path, Path]:
    """Create (or reuse) a stratified 80/10/10 split from a root of class folders."""
    split_root = source_root / "_prepared_splits"
    train_dir, val_dir, test_dir = split_root / "train", split_root / "val", split_root / "test"

    if (split_root / ".done").exists():
        return train_dir, val_dir, test_dir

    class_dirs = [
        p for p in sorted(source_root.iterdir())
        if p.is_dir() and not p.name.startswith((".", "_"))
    ]
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found under {source_root}")

    shutil.rmtree(split_root, ignore_errors=True)
    for class_dir in class_dirs:
        files = [p for p in class_dir.rglob("*")
                 if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        if len(files) < 3:
            continue
        train, rest = train_test_split(files, test_size=0.2, random_state=SEED)
        val, test = train_test_split(rest, test_size=0.5, random_state=SEED)
        for split_dir, subset in ((train_dir, train), (val_dir, val), (test_dir, test)):
            dst = split_dir / class_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for src in subset:
                shutil.copy2(src, dst / src.name)

    (split_root / ".done").touch()
    return train_dir, val_dir, test_dir


# --- Data location ---------------------------------------------------------
DATA_ROOT = Path(os.environ.get("CXR_DATA_ROOT", ROOT))
if (DATA_ROOT / "train").is_dir() and (DATA_ROOT / "val").is_dir() and (DATA_ROOT / "test").is_dir():
    TRAIN_DIR, VAL_DIR, TEST_DIR = DATA_ROOT / "train", DATA_ROOT / "val", DATA_ROOT / "test"
else:
    TRAIN_DIR, VAL_DIR, TEST_DIR = _split_class_dirs(DATA_ROOT)

# --- Hyper-parameters ------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = int(os.environ.get("CXR_BATCH_SIZE", 32))   # drop to 16/8 on a small GPU

EPOCHS_FROZEN, LR_FROZEN = 8, 1e-3       # phase 1: warm up the classifier head fast
EPOCHS_FINETUNE, LR_FINETUNE = 30, 1e-4  # phase 2: adapt the backbone to X-rays
FINE_TUNE_FROM = "conv4"                 # unfreeze DenseNet121's last two dense blocks (conv4 + conv5)
LABEL_SMOOTHING = 0.1                    # regularize noisy labels

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
BEST_MODEL_PATH = MODELS_DIR / "densenet121_chestxray.keras"
