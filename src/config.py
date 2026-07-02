"""Paths, classes and hyper-parameters (single place to tweak a run)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _split_dir(name: str) -> Path:
    """Find train/val/test, allowing a Google-Drive-style timestamp parent folder."""
    if (ROOT / name).is_dir():
        return ROOT / name
    return next((p / name for p in sorted(ROOT.glob("*")) if (p / name).is_dir()), ROOT / name)


TRAIN_DIR, VAL_DIR, TEST_DIR = _split_dir("train"), _split_dir("val"), _split_dir("test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16          # fits a 4 GB GPU (GTX 1650); raise if you have more VRAM
SEED = 42

EPOCHS_FROZEN, LR_FROZEN = 5, 1e-4         # phase 1: warmup head training (backbone frozen)
EPOCHS_FINETUNE, LR_FINETUNE = 25, 1e-5    # phase 2: fine-tune the backbone
FINE_TUNE_AT = 140                         # unfreeze top ~2 dense blocks
LABEL_SMOOTHING = 0.1                      # regularize noisy labels

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
BEST_MODEL_PATH = MODELS_DIR / "densenet121_chestxray.h5"
