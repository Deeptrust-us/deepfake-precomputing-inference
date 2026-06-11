# Quad-Stream Architecture (Audio)

This repo uses a **quad-stream** architecture specialized for **audio deepfake detection**.

## Overview

Each training/evaluation sample is built from one WAV file and one selected speech segment:

- **Segment STFT** (1, 224, 224)
- **Segment Log-Mel** (1, 224, 224)
- **Full-audio STFT** (1, 224, 224)
- **Full-audio Log-Mel** (1, 224, 224)

All inputs are **grayscale** (single-channel) time-frequency “images”.

## Input streams

```
Input (per sample):
  - segment_stft   (1, 224, 224) → segment_stft_stream   → (feature_dim,)
  - segment_logmel (1, 224, 224) → segment_logmel_stream → (feature_dim,)
  - full_stft      (1, 224, 224) → full_stft_stream      → (feature_dim,)
  - full_logmel    (1, 224, 224) → full_logmel_stream    → (feature_dim,)
```

## Feature extraction + fusion

- Each stream uses its **own backbone** (default: ResNet18) adapted to accept **1-channel** input.
- The 4 feature vectors are concatenated (optionally attention-weighted) then passed through an MLP head:

```
concat (4 * feature_dim)
  → fusion MLP
  → classifier (sigmoid) → p(fake)
```

## Preprocessing outputs (VAD segment selection)

Preprocessing runs VAD and generates:

- Full features once per file:
  - `features/full_stft/<stem>.npy`
  - `features/full_logmel/<stem>.npy`
- Segment features once per selected segment (`seg_num = 0,1,2,...`):
  - `features/segment_stft/<stem>_<seg_num>.npy`
  - `features/segment_logmel/<stem>_<seg_num>.npy`

Note: preprocessing can also write features for *all* detected segments (`--segment-mode all`), but the current dataset loader expects **exactly one** segment feature per stem.

## Dataset (`src/data/dataset.py`)

`DeepfakeDataset` is currently **file-level**: **one sample per `labels.json` entry**, pairing one segment feature pair with the full-audio feature pair:

- Segment pair: `<stem>_*.npy` (must match exactly one file)
- Full pair: `<stem>.npy`

Returned batch keys:

- `segment_stft`, `segment_logmel`, `full_stft`, `full_logmel`, `label`, `sample_id`, `filename`

## Model (`src/models/quad_stream.py`)

The model class is still named `QuadStreamModel` for compatibility, but it is an **audio quad-stream** model.

Forward signature:

```python
output = model(segment_stft, segment_logmel, full_stft, full_logmel)
```

All inputs are shaped `(B, 1, 224, 224)`.

## Training (`scripts/train.py`)

The training loop reads the 4 inputs from the batch dict and calls:

```python
outputs = model(segment_stft, segment_logmel, full_stft, full_logmel).squeeze()
```

Oversampling and label statistics in `scripts/train.py` are computed at the **file-sample** level (one label per dataset item).

