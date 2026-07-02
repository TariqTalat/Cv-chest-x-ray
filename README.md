# Chest X-ray Classification

Classifies chest X-ray images using transfer learning with **DenseNet121**.
The pipeline auto-detects the dataset classes from `train/`, `val/`, and
`test/` folders, so it adapts to the actual labels you provide.

## Project layout

```
.
├── main.py                 # run training + evaluation
├── pyproject.toml          # dependencies (managed by uv)
├── src/
│   ├── config.py           # paths, classes, hyper-parameters (edit here)
│   ├── data.py             # tf.data pipeline + augmentation + preprocessing
│   └── pipeline.py         # DenseNet121 model + 2-phase training + evaluation
├── models/                 # best model checkpoint (created on first run)
├── outputs/                # plots + reports (created on first run)
└── train/ val/ test/       # dataset (auto-detected, see below)
```

### Dataset
Point the pipeline at your data with the `CXR_DATA_ROOT` environment variable.
Two layouts are supported:

- **Already split** — a folder with `train/`, `val/`, `test/`, each holding one
  sub-folder per class:
  ```
  train/<ClassName>/*.png   val/<ClassName>/*.png   test/<ClassName>/*.png
  ```
- **Single root of class folders** — one sub-folder per class; the pipeline
  creates a stratified 80/10/10 split under `_prepared_splits/` on first run.

```powershell
$env:CXR_DATA_ROOT = "D:\data\chest-xray"   # override; defaults to the project root
```

## Setup

Uses [uv](https://docs.astral.sh/uv/). From the project root:

```powershell
uv sync
```

This creates `.venv` (Python 3.10, required by `tensorflow==2.10.1`) and
installs everything from `pyproject.toml`.

### GPU notes (important on Windows)
TensorFlow 2.10 is the **last** version with *native* Windows GPU support.
To actually use the GPU you need **CUDA 11.2** + **cuDNN 8.1** on PATH.
Easiest route is conda:

```powershell
conda create -n cxr python=3.10
conda activate cxr
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
uv pip install -e .
```

Without CUDA/cuDNN it runs on **CPU** (slower but works). Batch size defaults to
`32`; override with `CXR_BATCH_SIZE` and drop it to `16`/`8` on a small GPU
(e.g. a 4 GB GTX 1650). GPU memory-growth is enabled in `main.py`.

## Usage

```powershell
uv run python main.py
```

All settings — epochs, batch size, learning rates, fine-tuning depth — live in
[`src/config.py`](src/config.py). Edit there to change a run (e.g. lower
`EPOCHS_FROZEN`/`EPOCHS_FINETUNE` for a quick test, or `BATCH_SIZE` if you hit
out-of-memory).

## Outputs
- `models/densenet121_chestxray.keras` — best model (lowest validation loss)
- `outputs/training_history.png` — accuracy / loss / AUC curves
- `outputs/confusion_matrix.png` — test-set confusion matrix
- `outputs/classification_report.txt` — per-class precision/recall/F1

## Design notes (best practices)
- **Backbone + light head.** DenseNet121 → global average pool → dropout →
  softmax (the CheXNet baseline). No attention blocks or deep dense head — a
  small dataset can't support them without overfitting.
- **Correct preprocessing.** DenseNet `preprocess_input`, not `rescale=1./255`.
- **CXR-safe augmentation.** Small rotations (±10°), zoom, shift and mild
  intensity jitter via Keras layers. **No** 90° rotations or left/right flips —
  both are anatomically impossible for a chest film (a flip mislocates the
  heart, which matters for Cardiomegaly).
- **Two-phase training.** Frozen head → fine-tune only the last dense block,
  with backbone BatchNorm kept in inference mode.
- **One LR controller per phase.** ReduceLROnPlateau (warm-up) then cosine decay
  matched to the fine-tune length — no two schedulers fighting.
- **Sensible metrics.** Accuracy + macro AUC during training; per-class
  precision/recall/F1 from the test-set classification report.
- **Class weighting.** Inverse-frequency weights handle label imbalance.
- **Portable checkpoint.** Saved in the native `.keras` format.
