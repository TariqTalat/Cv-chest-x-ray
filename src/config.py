"""Paths, classes and hyper-parameters — the single place to tweak a run.

Data lives as one folder per class. Point at it by editing ``DATA_PARENT``
below (or set ``CXR_DATA_ROOT``); a stratified 80/10/10 split is created once
under the project's ``_prepared_splits/``.
"""
import os
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# --- Data location: folder that holds the three class sub-folders ----------
CLASS_NAMES = ["Cardiomegaly", "Nodule_Mass", "Normal"]
DATA_PARENT = Path(os.environ.get("CXR_DATA_ROOT", "/home/aiteam/Tarek/side projects"))
CLASS_DIRS = [DATA_PARENT / name for name in CLASS_NAMES]


def _prepare_splits(class_dirs) -> tuple[Path, Path, Path]:
    """Create (or reuse) a stratified 80/10/10 split from a list of class folders."""
    split_root = ROOT / "_prepared_splits"
    train_dir, val_dir, test_dir = split_root / "train", split_root / "val", split_root / "test"

    if (split_root / ".done").exists():
        return train_dir, val_dir, test_dir

    shutil.rmtree(split_root, ignore_errors=True)
    split_root.mkdir(parents=True, exist_ok=True)

    total = 0
    for class_dir in class_dirs:
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        files = [p for p in class_dir.rglob("*")
                 if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        if len(files) < 3:
            raise FileNotFoundError(f"Need >=3 images in {class_dir}, found {len(files)}")
        train, rest = train_test_split(files, test_size=0.2, random_state=SEED)
        val, test = train_test_split(rest, test_size=0.5, random_state=SEED)
        for split_dir, subset in ((train_dir, train), (val_dir, val), (test_dir, test)):
            dst = split_dir / class_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for src in subset:
                shutil.copy2(src, dst / src.name)
        total += len(files)

    print(f"Prepared {total} images into {split_root}")
    (split_root / ".done").touch()
    return train_dir, val_dir, test_dir


TRAIN_DIR, VAL_DIR, TEST_DIR = _prepare_splits(CLASS_DIRS)

# --- Hyper-parameters ------------------------------------------------------
IMG_SIZE = (320, 320)                    # bigger input = more pixels on small nodules/masses
BATCH_SIZE = int(os.environ.get("CXR_BATCH_SIZE", 32))   # drop to 16/8 on a small GPU

EPOCHS_FROZEN, LR_FROZEN = 8, 1e-3       # phase 1: warm up the classifier head fast
EPOCHS_FINETUNE, LR_FINETUNE = 30, 1e-4  # phase 2: adapt the backbone to X-rays
FINE_TUNE_FROM = "conv3"                 # unfreeze DenseNet121's last three dense blocks (conv3-5)
FOCAL_GAMMA = 2.0                        # focal loss: focus training on hard cases (subtle nodules)
CLASS_WEIGHT_BOOST = {"Nodule_Mass": 2.0}  # over-weight the under-predicted class to lift its recall

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
BEST_MODEL_PATH = MODELS_DIR / "densenet121_chestxray.keras"
