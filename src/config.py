"""Paths, classes and hyper-parameters (single place to tweak a run).

Point the pipeline at your data with the ``CXR_DATA_ROOT`` env var. It may be
either a folder already split into ``train/``, ``val/`` and ``test/``, or a
single folder containing one sub-folder per class (an 80/10/10 split is then
created automatically under ``_prepared_splits/``).
"""
import os
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _prepare_split_dirs(source_root: Path) -> tuple[Path, Path, Path]:
    """Create a stratified train/val/test split from a root of class folders."""
    split_root = source_root / "_prepared_splits"
    train_dir = split_root / "train"
    val_dir = split_root / "val"
    test_dir = split_root / "test"

    if all((d / "_placeholder").exists() for d in (train_dir, val_dir, test_dir)):
        return train_dir, val_dir, test_dir

    shutil.rmtree(split_root, ignore_errors=True)
    for split_dir in (train_dir, val_dir, test_dir):
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "_placeholder").touch()

    class_dirs = [
        path
        for path in sorted(source_root.iterdir())
        if path.is_dir()
        and path.name not in {"train", "val", "test", "_prepared_splits", ".venv", "chest-project"}
        and not path.name.startswith(".")
    ]
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found under {source_root}")

    for class_dir in class_dirs:
        files = [
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not files:
            continue

        train_files, temp_files = train_test_split(files, test_size=0.2, random_state=SEED)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=SEED)

        for target_dir, file_list in ((train_dir, train_files), (val_dir, val_files), (test_dir, test_files)):
            target_class_dir = target_dir / class_dir.name
            target_class_dir.mkdir(parents=True, exist_ok=True)
            for src_path in file_list:
                shutil.copy2(src_path, target_class_dir / src_path.name)

    return train_dir, val_dir, test_dir


# Data location: override with `CXR_DATA_ROOT`; defaults to the project root.
DATA_ROOT = Path(os.environ.get("CXR_DATA_ROOT", ROOT))
if (DATA_ROOT / "train").is_dir() and (DATA_ROOT / "val").is_dir() and (DATA_ROOT / "test").is_dir():
    TRAIN_DIR, VAL_DIR, TEST_DIR = DATA_ROOT / "train", DATA_ROOT / "val", DATA_ROOT / "test"
else:
    TRAIN_DIR, VAL_DIR, TEST_DIR = _prepare_split_dirs(DATA_ROOT)

IMG_SIZE = (224, 224)
BATCH_SIZE = int(os.environ.get("CXR_BATCH_SIZE", 32))   # drop to 16/8 on a small GPU

EPOCHS_FROZEN, LR_FROZEN = 15, 1e-4         # phase 1: warm up the classifier head
EPOCHS_FINETUNE, LR_FINETUNE = 40, 1e-5     # phase 2: fine-tune the backbone
FINE_TUNE_AT = 313                          # unfreeze only DenseNet121's last dense block (conv5)
LABEL_SMOOTHING = 0.1                       # regularize noisy labels

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
BEST_MODEL_PATH = MODELS_DIR / "densenet121_chestxray.keras"   # native Keras format (portable reload)
