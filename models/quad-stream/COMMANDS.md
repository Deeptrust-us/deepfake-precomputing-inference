# Command Reference (Audio Quad-Stream)

Quick reference for the **audio quad-stream** pipeline.

> All commands assume you run them from the repo root and you activate the environment using:
>
> `source venv/bin/active`

## Pipeline (recommended order)

```bash
source venv/bin/active

# 1) Preprocess: WAVs → VAD segment selection → features/*.npy
#
# IMPORTANT: The current training/eval dataset loader expects EXACTLY ONE segment feature
# per audio file (it will error if multiple <stem>_<seg_num>.npy exist).
# Use --segment-mode random_one unless you know you want multi-segment outputs for analysis.
python scripts/preprocess_audio.py --dataset-root /path/to/dataset_audios --segment-mode random_one --seed 42
# Disk-saving option: also delete source WAVs after feature extraction (irreversible)
python scripts/preprocess_audio.py --dataset-root /path/to/dataset_audios --segment-mode random_one --seed 42 --delete-audio
# Faster on CPU: parallelize across files (tune workers/threads for your machine)
python scripts/preprocess_audio.py --dataset-root /path/to/dataset_audios --num-workers 8 --torch-threads 1

# 2) Train
#
# Training reads the dataset path from config: data.data_root
# Edit config/config.yaml (recommended) or create your own config file.
python scripts/train.py --config config/config.yaml

# 3) Evaluate
python scripts/evaluate.py --config config/config.yaml --checkpoint checkpoints/best_model.pth --split test
```

## Preprocessing

### Basic

```bash
source venv/bin/active
python scripts/preprocess_audio.py --dataset-root /path/to/dataset_audios --segment-mode random_one --seed 42
```

### Disk-saving: one random segment per WAV

```bash
source venv/bin/active
python scripts/preprocess_audio.py --dataset-root /path/to/dataset_audios --segment-mode random_one --seed 42
```

### Disk-saving: delete source WAVs after feature extraction (irreversible)

```bash
source venv/bin/active
python scripts/preprocess_audio.py --dataset-root /path/to/dataset_audios --segment-mode random_one --seed 42 --delete-audio
```

### Overwrite existing features

```bash
source venv/bin/active
python scripts/preprocess_audio.py --dataset-root /path/to/dataset_audios --overwrite
```

### Preprocessing options

```bash
source venv/bin/active
python scripts/preprocess_audio.py \
  --dataset-root /path/to/dataset_audios \
  --sample-rate 16000 \
  --segment-seconds 2.0
```

## Training

### Basic training

```bash
source venv/bin/active
python scripts/train.py --config config/config.yaml
```

### Resume from checkpoint

```bash
source venv/bin/active
python scripts/train.py --config config/config.yaml --resume checkpoints/latest.pth
```

### Use explicit split label files (optional)

If your `data.data_root` contains split files like `labels_train.json` / `labels_val.json`, the training script will auto-pick them.
If you want to force specific split label files, set them in `config/config.yaml`:

```yaml
data:
  train_labels_file: "/path/to/data_root/labels_train.json"
  val_labels_file: "/path/to/data_root/labels_val.json"
  test_labels_file: "/path/to/data_root/labels_test.json"
paths:
  features_dir: "/path/to/data_root/features"
```

## Evaluation

```bash
source venv/bin/active
python scripts/evaluate.py --config config/config.yaml --checkpoint checkpoints/best_model.pth --split test
```

## Monitoring

### TensorBoard

```bash
tensorboard --logdir logs
```

## Dataset sanity checks

```bash
source venv/bin/active
python -c "from src.data.dataset import DeepfakeDataset; ds=DeepfakeDataset('/path/to/dataset_audios'); print('Samples:', len(ds)); print('First keys:', sorted(ds[0].keys()))"
```

Check feature folders:

```bash
ls -1 /path/to/dataset_audios/features/segment_stft | head
ls -1 /path/to/dataset_audios/features/full_stft | head
```

## Dataset utilities (optional)

### Create an 8-way split (A/B halves + train/val/test)

```bash
source venv/bin/active
python scripts/utils/split_dataset.py --labels-path data/labels.json --out-dir data/splits_8 --seed 42
```

## Quick reference table

| Task | Command |
|------|---------|
| Preprocess | `python scripts/preprocess_audio.py --dataset-root /path/to/dataset_audios --segment-mode random_one --seed 42` |
| Train | `python scripts/train.py --config config/config.yaml` |
| Evaluate | `python scripts/evaluate.py --config config/config.yaml --checkpoint checkpoints/best_model.pth --split test` |
| TensorBoard | `tensorboard --logdir logs` |
