# Design rationale — best practices for chest X-ray classification

This project uses a strong pretrained backbone with a light classifier head and
anatomy-preserving augmentation. That is the safest strategy for a small
medical dataset, where a heavy "enhanced" head or impossible transforms usually
hurt generalization.

---

## 1. Architecture (`src/pipeline.py`)

**DenseNet121 (ImageNet) → GlobalAveragePooling2D → Dropout(0.4) → Dense(softmax).**

This is the proven CheXNet-style baseline. The model avoids additional
attention/SE blocks, dual pooling, and multi-layer funnel heads because those
add parameters and overfitting risk without clear benefit on a 3-class CXR task.

- Backbone is built with `training=False` so BatchNorm layers preserve their
  pretrained running statistics.
- Fine-tuning unfreezes the backbone's **last two dense blocks**
  (`FINE_TUNE_FROM = "conv4"`) at LR `1e-4`, so the features actually adapt to
  X-rays instead of staying near their ImageNet state.

## 2. Data augmentation (`src/data.py`)

Augmentation is intentionally anatomy-preserving:

- Rotation ±10°, zoom ±10%, translation ±5%, mild contrast and brightness jitter.
- **No 90° rotations** — a sideways/upside-down chest X-ray is anatomically
  impossible.
- **No horizontal flip** — mirroring a chest X-ray moves the heart and mediastinum
  to the wrong side, which is dangerous for laterality-sensitive labels like
  Cardiomegaly.

The intensity jitter is applied on the original 0–255 image scale, so the
brightness delta is actually meaningful.

## 3. Training (`src/train.py`)

- **Two-phase training:** warm up the classifier head with the backbone frozen,
  then fine-tune the backbone.
- **One LR controller per phase:** ReduceLROnPlateau in phase 1 and cosine decay
  in phase 2. The cosine schedule is matched to the actual fine-tune horizon.
- **Metrics:** track accuracy and macro AUC (`multi_label=True`). Per-class
  precision/recall/F1 are computed from the evaluation report, not from
  thresholded softmax metrics.
- **Checkpointing and early stopping** both monitor `val_loss`, so the saved
  model and restored weights are consistent.
- **Class weighting** uses inverse-frequency weights from `src/data.py`.

## 4. Config (`src/config.py`)

- Dataset root is controlled via `CXR_DATA_ROOT`.
- `BATCH_SIZE` can be overridden with `CXR_BATCH_SIZE`.
- The best model is saved as `models/densenet121_chestxray.keras`.

---

## Next steps

1. Add Grad-CAM visualizations to verify the model focuses on relevant chest
   anatomy rather than spurious text or borders.
2. Add cross-validation or repeated runs if you need more robustness on a small
   dataset.
3. Keep the model simple until you have enough images to justify additional
   architectural complexity.
