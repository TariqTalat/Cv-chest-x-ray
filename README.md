# Chest X-ray Classification (4 classes)

Classifies chest X-ray images into **Edema**, **Nodule Mass**, **Normal**,
**Pneumonia** using transfer learning with **DenseNet121** (the backbone used
by *CheXNet*, the reference model for NIH chest X-rays).

## Project layout

```
.
├── main.py                 # run training + evaluation
├── pyproject.toml          # dependencies (managed by uv)
├── src/
│   ├── config.py           # paths, classes, hyper-parameters (edit here)
│   ├── data.py             # tf.data pipeline + augmentation + preprocessing
│   ├── model.py            # DenseNet121 backbone + classifier head
│   ├── train.py            # 2-phase training + training curves
│   └── evaluate.py         # accuracy, classification report, confusion matrix
├── models/                 # best model checkpoint (created on first run)
├── outputs/                # plots + reports (created on first run)
└── train/ val/ test/       # dataset (auto-detected, see below)
```

### Dataset
Each split folder contains one sub-folder per class:

```
train/<ClassName>/*.png      val/<ClassName>/*.png      test/<ClassName>/*.png
```

The split folders are **auto-detected**, so the messy Google-Drive export
names (e.g. `train-20260620T...-001/train`) work as-is — no need to rename.

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

Without CUDA/cuDNN it runs on **CPU** (slower but works). The 4 GB GTX 1650
is handled by `BATCH_SIZE=16` in `config.py` and GPU memory-growth in `main.py`;
drop the batch size to 8 if you hit out-of-memory.

## Usage

```powershell
uv run python main.py
```

All settings — epochs, batch size, learning rates, fine-tuning depth — live in
[`src/config.py`](src/config.py). Edit there to change a run (e.g. lower
`EPOCHS_FROZEN`/`EPOCHS_FINETUNE` for a quick test, or `BATCH_SIZE` if you hit
out-of-memory).

## Outputs
- `models/densenet121_chestxray.h5` — best model (by validation accuracy)
- `outputs/training_history.png` — accuracy / loss curves
- `outputs/confusion_matrix.png` — test-set confusion matrix
- `outputs/classification_report.txt` — per-class precision/recall/F1

## What changed vs. the original notebook
- One clean, reproducible architecture (DenseNet121) instead of two competing
  ones; dropped the heavier DenseNet+ResNet fusion to keep it focused.
- Correct DenseNet `preprocess_input` instead of `rescale=1./255` (matters for
  ImageNet transfer learning).
- Two-phase training (frozen head → fine-tune) with BatchNorm kept frozen.
- Proper test evaluation: classification report + confusion matrix, not just accuracy.
- The one-off `train_test_split` copy step is removed — the data is already split.
