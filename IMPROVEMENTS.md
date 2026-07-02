# Design rationale — best practices for 3-class chest X-ray classification

This documents *why* the pipeline is built the way it is. The guiding principle:
a strong pretrained backbone with a light head, plus **anatomy-preserving**
augmentation, beats a heavy "enhanced" architecture on a small medical dataset.

---

## 1. Architecture (`src/model.py`)

**DenseNet121 (ImageNet) → GlobalAveragePooling2D → Dropout(0.4) → Dense(softmax).**

This is essentially the CheXNet design. We deliberately do **not** add
squeeze-excitation / spatial-attention blocks, dual (avg+max) pooling, or a deep
`1024→512→256` funnel head. On a few-thousand-image, 3-class set those add
millions of trainable parameters and overfit without a reliable accuracy gain.

- Backbone is called with `training=False` so its BatchNorm layers keep the
  ImageNet running statistics — the recommended setup when fine-tuning on a
  small dataset.
- Fine-tuning unfreezes only the **last dense block** (`FINE_TUNE_AT = 313`),
  not ~two-thirds of the network, to avoid catastrophic forgetting.

## 2. Data augmentation (`src/data.py`)

CXR-safe transforms only, via Keras preprocessing layers (which scale correctly
for a 0–255 image, unlike hand-rolled ops where a ±0.1 brightness delta is
invisible):

- Rotation ±10°, zoom ±10%, translation ±5%, mild contrast/brightness jitter.
- **No 90° rotations** — a sideways chest film never occurs.
- **No left/right flip** — mirroring moves the heart to the right side, which is
  anatomically wrong and directly harmful for a class like Cardiomegaly.

## 3. Training (`src/train.py`)

- **Two phases:** warm up the head (backbone frozen) → fine-tune the last block.
- **One LR controller per phase:** ReduceLROnPlateau in phase 1, cosine decay in
  phase 2 — matched to the actual fine-tune length. Never both at once.
- **Loss:** categorical cross-entropy with label smoothing (0.1).
- **Metrics:** accuracy + macro AUC (`multi_label=True`). Per-class precision /
  recall / F1 come from the test-set classification report, not the built-in
  Precision/Recall metrics (which threshold a softmax at 0.5 and are meaningless
  for >2 classes).
- **Checkpoint + early stopping** both track `val_loss` for a consistent "best".
- **Class weights:** inverse-frequency, to handle label imbalance.

## 4. Config (`src/config.py`)

- Data location via `CXR_DATA_ROOT` (no hardcoded paths); accepts a pre-split
  dataset or a single root of class folders (auto 80/10/10 stratified split).
- `BATCH_SIZE` via `CXR_BATCH_SIZE` (default 32).
- Model saved in the portable native `.keras` format.

---

## Optional next steps

1. **Grad-CAM** overlays — confirm the model attends to the heart / nodule, not
   to text markers or borders. High value for a medical model.
2. **Test-time augmentation** — average predictions over a few augmented views.
3. **Ensemble / cross-validation** — if you need to squeeze out more robustness.

## References
- CheXNet — https://arxiv.org/abs/1711.05225
- Keras transfer-learning & fine-tuning guide — https://keras.io/guides/transfer_learning/
- Anatomy-aware CXR augmentation — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12194474/
