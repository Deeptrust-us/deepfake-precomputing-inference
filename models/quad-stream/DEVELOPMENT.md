# Development Guide (Audio Quad-Stream)

This guide covers development workflows for the **audio** quad-stream pipeline.

## Environment setup

From the repo root:

```bash
source venv/bin/active
python -c "import torch, torchaudio; print('OK')"
```

## Core pipeline commands

```bash
source venv/bin/active

# Preprocess
python scripts/preprocess_audio.py --dataset-root /path/to/dataset_root --segment-mode random_one --seed 42

# Train
python scripts/train.py --config config/config.yaml

# Evaluate
python scripts/evaluate.py --config config/config.yaml --checkpoint checkpoints/best_model.pth --split test
```

## Code organization (relevant modules)

- `scripts/preprocess_audio.py`
  - Reads `labels.json`, resolves WAV paths, runs VAD, writes `features/*.npy`.
  - Use `--segment-mode random_one` for training/eval compatibility (one segment per file).

- `src/utils/audio_features.py`
  - WAV loading/resampling
  - VAD segment detection
  - STFT and Log-Mel generation
  - Resize/normalize to `(1,224,224)`

- `src/data/dataset.py`
  - Builds **one sample per labels.json entry** (file-level dataset)
  - Loads exactly one segment pair + one full pair per stem

- `src/models/quad_stream.py`
  - `QuadStreamModel` (name kept for compatibility)
  - Audio quad-stream forward signature:
    - `model(segment_stft, segment_logmel, full_stft, full_logmel)`

## Debugging

### Verify preprocessing outputs exist

```bash
ls -1 /path/to/dataset_root/features/segment_stft | head
ls -1 /path/to/dataset_root/features/full_stft | head
```

### Check dataset loading

```bash
source venv/bin/active
python - <<'PY'
from src.data.dataset import DeepfakeDataset

ds = DeepfakeDataset('/path/to/dataset_root')
print('samples:', len(ds))
item = ds[0]
print('keys:', sorted(item.keys()))
print('sample_id:', item['sample_id'])
print('filename:', item['filename'])
print('shapes:', item['segment_stft'].shape, item['full_logmel'].shape)
PY
```

### Check model forward pass

```bash
source venv/bin/active
python - <<'PY'
import torch
from src.models.quad_stream import QuadStreamModel

model = QuadStreamModel(backbone='resnet18', feature_dim=64, fusion_dim=128, pretrained=False)
model.eval()

x = torch.randn(2, 1, 224, 224)
with torch.no_grad():
    y = model(x, x, x, x)
print('output:', y.shape)
PY
```

## Notes

- The repo contains legacy video-related utilities and docs from the previous implementation. The active training/eval pipeline is **audio-only**.
