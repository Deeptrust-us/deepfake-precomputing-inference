# Quick Start Guide (Audio Quad-Stream)

This guide gets you running the **audio quad-stream** pipeline end-to-end:

WAVs + `labels.json` → (one selected speech segment + full audio) → 4 feature “images” per sample → train/eval.

## Prerequisites

- Python 3.10+ recommended
- A working virtual environment under `venv/`

## Step 1: Setup

```bash
source venv/bin/active
python -c "import torch, torchaudio; print('OK')"
```

## Step 2: Prepare your dataset directory

Required:

```
dataset_audios/
├── labels.json
└── audio/
    ├── 0000000001.wav              # Option A (preferred)
    └── real/0000000001.wav         # Option B (fallback)
       fake/0000001234.wav
```

`labels.json` must be a list of entries with at minimum:

- `filename`: wav filename (e.g. `"0000000001.wav"`)
- `label`: `"real"` or `"fake"`

## Step 3: Preprocess (generate features)

This generates:

- `features/full_{stft,logmel}/<stem>.npy` once per WAV
- `features/segment_{stft,logmel}/<stem>_<seg_num>.npy` for the selected segment

```bash
source venv/bin/active
python scripts/preprocess_audio.py --dataset-root /path/to/dataset_audios --segment-mode random_one --seed 42
```

Recommended option (writes **exactly one** segment per WAV; still writes `full_*` once per WAV):

```bash
source venv/bin/active
python scripts/preprocess_audio.py --dataset-root /path/to/dataset_audios --segment-mode random_one --seed 42
```

Disk-saving option (also delete source WAVs after feature extraction; **irreversible**):

```bash
source venv/bin/active
python scripts/preprocess_audio.py --dataset-root /path/to/dataset_audios --segment-mode random_one --seed 42 --delete-audio
```

Re-generate even if files exist:

```bash
source venv/bin/active
python scripts/preprocess_audio.py --dataset-root /path/to/dataset_audios --overwrite
```

## Step 4: Train

Training reads the dataset path from `config/config.yaml` (`data.data_root`).
Set it to your dataset directory (e.g. `/path/to/dataset_audios`) before running.

```bash
source venv/bin/active
python scripts/train.py --config config/config.yaml
```

Resume training:

```bash
source venv/bin/active
python scripts/train.py --config config/config.yaml --resume checkpoints/latest.pth
```

## Step 5: Evaluate

```bash
source venv/bin/active
python scripts/evaluate.py --config config/config.yaml --checkpoint checkpoints/best_model.pth --split test
```

## Troubleshooting

### “No module named src”

Run commands from the repo root (same folder as `scripts/` and `src/`).

### “Missing precomputed feature(s) …”

Run preprocessing first:

```bash
source venv/bin/active
python scripts/preprocess_audio.py --dataset-root /path/to/dataset_audios --segment-mode random_one --seed 42
```
