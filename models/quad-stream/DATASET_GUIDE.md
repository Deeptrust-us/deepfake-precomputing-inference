# Dataset Guide (Audio Quad-Stream)

This project trains and evaluates an **audio quad-stream** deepfake detector using WAV files described by a `labels.json`.

Preprocessing runs **VAD** (Silero if available, otherwise a whole-audio fallback) to select speech and generates **4 time-frequency “images”** per audio file:

- **Segment** features: STFT + Log-Mel (one selected speech segment)
- **Full-audio** features: STFT + Log-Mel (whole file)

**Current behavior (important):** the shipped `DeepfakeDataset` is **file-level**: **one dataset sample per `labels.json` entry**. That means your dataset must contain **exactly one** segment feature pair per file stem.

## Supported dataset layout

### Required

```
<dataset_root>/
├── labels.json
└── audio/
```

### Audio file layouts

- **Option A (preferred)**: flat audio folder

```
<dataset_root>/
├── labels.json
└── audio/
    ├── 0000000001.wav
    ├── 0000000002.wav
    └── ...
```

- **Option B (fallback)**: organized real/fake folders

```
<dataset_root>/
├── labels.json
└── audio/
    ├── real/
    │   ├── 0000000001.wav
    │   └── ...
    └── fake/
        ├── 0000001234.wav
        └── ...
```

## labels.json format

`labels.json` must be a **JSON list**. Each entry must include:

- `filename`: WAV filename (e.g. `"0000000001.wav"`)
- `label`: `"real"` or `"fake"`

Optional fields (kept and returned by the dataset):

- `language`
- `model_or_speaker`

Example:

```json
[
  {"filename": "0000000001.wav", "label": "real", "language": "en", "model_or_speaker": "speakerA"},
  {"filename": "0000000002.wav", "label": "fake", "language": "en", "model_or_speaker": "modelX"}
]
```

## Preprocessing outputs

Preprocessing writes the following folders under `<dataset_root>/features/`:

- **Full audio (saved once per WAV)**
  - `features/full_stft/<stem>.npy`
  - `features/full_logmel/<stem>.npy`

- **Speech segment (saved once per selected VAD segment)**
  - `features/segment_stft/<stem>_<seg_num>.npy`
  - `features/segment_logmel/<stem>_<seg_num>.npy`

Where:

- `<stem> = Path(filename).stem`
- `<seg_num>` is the VAD segment index: `0, 1, 2, ...`

## Run preprocessing

```bash
source venv/bin/active
python scripts/preprocess_audio.py --dataset-root /path/to/dataset_root --segment-mode random_one --seed 42
```

Recommended option (writes **exactly one** segment per WAV; still writes `full_*` once per WAV):

```bash
source venv/bin/active
python scripts/preprocess_audio.py --dataset-root /path/to/dataset_root --segment-mode random_one --seed 42
```

Disk-saving option (also delete source WAVs after feature extraction; **irreversible**):

```bash
source venv/bin/active
python scripts/preprocess_audio.py --dataset-root /path/to/dataset_root --segment-mode random_one --seed 42 --delete-audio
```

Overwrite previously generated files:

```bash
source venv/bin/active
python scripts/preprocess_audio.py --dataset-root /path/to/dataset_root --overwrite
```

### Optional: generate features for *all* detected segments

You can ask preprocessing to write one `segment_*` pair per detected segment:

```bash
source venv/bin/active
python scripts/preprocess_audio.py --dataset-root /path/to/dataset_root --segment-mode all
```

However, **training/evaluation with `DeepfakeDataset` will fail** if multiple `segment_*` files exist for the same stem. Use this mode only for analysis, or if you plan to select/keep one segment per file afterward.

## How training samples are formed

The dataset is **file-based**:

- One sample corresponds to one `labels.json` entry (one audio file stem).
- Segment features are loaded by matching exactly one file:
  - `segment_stft/<stem>_*.npy` (must match **exactly one** file)
  - `segment_logmel/<stem>_*.npy` (must match **exactly one** file)
- Full features are loaded from:
  - `full_stft/<stem>.npy`
  - `full_logmel/<stem>.npy`
- The label is taken from the WAV’s `labels.json` entry.

## Recommendations

- **Unique stems**: avoid two different entries that would share the same `<stem>`.
- **Start with a tiny dataset** (2–10 WAVs) to validate preprocessing and training.
- If your dataset has long WAVs, tune preprocessing with:
  - `--segment-seconds` (controls per-segment crop/pad length)
  - `--sample-rate`

## Troubleshooting

### WAV not found

Ensure the WAV exists under `<dataset_root>/audio/` following Option A or Option B and that `filename` matches exactly.

### No segment features found

Run preprocessing and verify that files exist under:

- `features/segment_stft/`
- `features/segment_logmel/`

### “Expected exactly one segment feature … found N”

You likely ran preprocessing with `--segment-mode all`, which produces multiple `segment_*` files per stem.
Re-run preprocessing with `--segment-mode random_one` (or delete extra `segment_*` files so only one remains per stem).
