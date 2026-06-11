# Dataset Analysis (Audio Quad-Stream)

This document explains how to reason about dataset size and class balance in the **audio quad-stream** pipeline.

The current training/evaluation pipeline is **file-level**:

- 1 WAV file (one `labels.json` entry) → 1 dataset sample
- Each sample uses **one selected speech segment** + **full-audio** features

So the effective dataset size is driven by **how many files** you have (not by VAD segment counts).

## What preprocessing produces

For each `labels.json` entry (`filename`, `label`):

- Full features (once per file)
  - `features/full_stft/<stem>.npy`
  - `features/full_logmel/<stem>.npy`

- Segment features (one pair for the selected segment)
  - `features/segment_stft/<stem>_<seg_num>.npy`
  - `features/segment_logmel/<stem>_<seg_num>.npy`

If you run preprocessing with `--segment-mode all`, you will get **multiple** `segment_*` files per stem (one per detected segment). This is useful for analysis, but is **not compatible** with the current `DeepfakeDataset` loader.

## Useful statistics to compute

### 1) File counts

- Total WAV files referenced by `labels.json`
- Missing WAV files on disk

### 2) Feature completeness (what training/eval actually needs)

- Missing `full_*` feature files
- Stems with **0** or **>1** segment feature matches (`segment_*` must match exactly one file per stem)

### 3) Class balance (file-level)

Labels are assigned per file, so class balance is measured by counting `labels.json` entries (`real` vs `fake`).

If you generated features with `--segment-mode all` for analysis, you can also look at segment counts per file, but training/eval do not use those extra segments.

## How to compute stats (example script)

Run this from repo root:

```bash
source venv/bin/active
python - <<'PY'
import json
from pathlib import Path
from collections import Counter

dataset_root = Path('/path/to/dataset_root')
labels = json.loads((dataset_root/'labels.json').read_text())

# file-level
real_files = sum(1 for e in labels if str(e.get('label','')).lower()=='real')
fake_files = sum(1 for e in labels if str(e.get('label','')).lower()=='fake')
print('files:', len(labels), 'real:', real_files, 'fake:', fake_files)

# feature completeness
features = dataset_root/'features'
full_stft = features/'full_stft'
full_logmel = features/'full_logmel'
seg_stft = features/'segment_stft'
seg_logmel = features/'segment_logmel'

stems = [Path(e['filename']).stem for e in labels if e.get('filename')]
missing_full = [s for s in stems if not (full_stft/f'{s}.npy').exists() or not (full_logmel/f'{s}.npy').exists()]
print('missing full features:', len(missing_full))

# segment matches per stem (must be exactly one for training/eval)
seg_counts = Counter()
for s in stems:
    seg_counts[s] = len(list(seg_stft.glob(f'{s}_*.npy')))

zero = sum(1 for s, n in seg_counts.items() if n == 0)
one = sum(1 for s, n in seg_counts.items() if n == 1)
many = sum(1 for s, n in seg_counts.items() if n > 1)
print('segment feature matches per stem:', 'zero:', zero, 'one:', one, 'many:', many)

if many:
    # If you used --segment-mode all, this helps you understand the distribution.
    import statistics as st
    vals = sorted(seg_counts.values())
    print('segment files per stem distribution:')
    print('  min:', vals[0], 'median:', st.median(vals), 'mean:', sum(vals)/len(vals), 'max:', vals[-1])
PY
```

## Sanity checks

### Feature shapes

All saved features are expected to be **float32** arrays with shape:

- `(1, 224, 224)`

Quick check:

```bash
source venv/bin/active
python - <<'PY'
import numpy as np
from pathlib import Path
p = Path('/path/to/dataset_root/features/full_logmel')
ex = next(p.glob('*.npy'))
x = np.load(ex)
print(ex.name, x.shape, x.dtype)
PY
```

### Dataset loading

```bash
source venv/bin/active
python - <<'PY'
from src.data.dataset import DeepfakeDataset

ds = DeepfakeDataset('/path/to/dataset_root')
print('samples:', len(ds))
print('keys:', sorted(ds[0].keys()))
print('shapes:', ds[0]['segment_stft'].shape, ds[0]['full_logmel'].shape)
PY
```
