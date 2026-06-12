# Deepfake Precomputing Inference

This repository precomputes classifier outputs for a deepfake audio dataset using two independent models:

- **hm-conformer** — raw waveform inference (LFCC + HM-Conformer)
- **quad-stream** — precomputed feature inference (4-stream ResNet fusion)

The scripts follow the output contract in [`docs/inference-results-specification.md`](docs/inference-results-specification.md).

## Repository layout

```text
data/
  labels.json              # Dataset metadata (sample ids, paths, labels)
  raw/                     # Audio files and/or precomputed features (not bundled)
models/
  hm-conformer/            # HM-Conformer model code and checkpoints
  quad-stream/             # QuadStream model code and checkpoints
scripts/
  run_inference.py         # Run one model over labels.json
  merge_results.py         # Optional merge step after both models finish
docs/
  inference-results-specification.md
tests/                     # Unit tests (no dataset download required)
```

## Environment setup

Create and activate a local virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

After adding new packages, freeze the environment:

```bash
pip freeze > requirements.txt
```

## Dataset metadata

Inference reads sample metadata from `data/labels.json`. Each entry looks like:

```json
{
  "id": "0000556524",
  "filename": "0000556524.wav",
  "original_path": "/path/to/source.wav",
  "label": "real",
  "language": "en",
  "model_or_speaker": "speaker_name"
}
```

The loader maps this to the spec schema:

| labels.json field | samples.parquet column |
|---|---|
| `id` | `sample_id` |
| `filename` | `filename` |
| `original_path` | `original_path` |
| `label` (`real`/`fake`) | `ground_truth` (`real`/`deepfake`) |
| `language` | `language` |
| `model_or_speaker` | `manipulation_type` (`none` for real samples) |

## Prerequisites before running inference

### HM-Conformer

1. Place checkpoints under `models/hm-conformer/params/` (or pass `--checkpoint-path`).
2. Place audio files under `data/raw/` using the filenames from `labels.json`.
   - Resolution order: `data/raw/<filename>`, `data/raw/audio/<filename>`, then `original_path` if it exists locally.

Expected checkpoint files:

```text
models/hm-conformer/params/
  check_point_DF_frontend_20.pt
  check_point_DF_backend0_20.pt
  ...
  check_point_DF_loss4_20.pt
```

### QuadStream

1. Place the trained checkpoint, e.g. `models/quad-stream/checkpoints/best_model.pth`.
2. Precompute features with the quad-stream preprocessing pipeline:

```bash
cd models/quad-stream
python scripts/preprocess_audio.py \
  --dataset-root ../../data/raw \
  --segment-mode random_one --seed 42
```

3. Ensure features exist under `data/raw/features/`:

```text
data/raw/features/
  segment_stft/
  segment_logmel/
  full_stft/
  full_logmel/
```

## Run inference

Each model runs independently and writes its own timestamped run folder.

### HM-Conformer

```bash
python scripts/run_inference.py \
  --model-name hm-conformer \
  --dataset-root data/raw \
  --metadata-path data/labels.json \
  --checkpoint-path models/hm-conformer/params \
  --output-dir data/inference/hm-conformer
```

### QuadStream

```bash
python scripts/run_inference.py \
  --model-name quad-stream \
  --dataset-root data/raw \
  --metadata-path data/labels.json \
  --checkpoint-path models/quad-stream/checkpoints/best_model.pth \
  --features-dir data/raw/features \
  --output-dir data/inference/quad-stream \
  --trust-checkpoint
```

Useful flags:

| Flag | Description |
|---|---|
| `--limit N` | Process only the first N samples (smoke test) |
| `--skip-missing` | Skip samples with missing audio/features instead of failing |
| `--debug` | Print per-stage timing to stderr |
| `--batch-size N` | Quad-stream batch size (default from model config) |
| `--num-workers N` | Quad-stream DataLoader workers (default: auto) |
| `--cpu-threads N` | PyTorch/BLAS threads for quad-stream CPU inference |
| `--prefetch-factor N` | Quad-stream DataLoader prefetch depth per worker |
| `--run-id ID` | Override the default `YYYY-MM-DD_<model-name>` folder name |

Example output:

```text
data/inference/
├── hm-conformer/
│   └── 2026-06-11_hm-conformer/
│       ├── samples.parquet
│       ├── results.parquet
│       └── manifest.json
└── quad-stream/
    └── 2026-06-11_quad-stream/
        ├── samples.parquet
        ├── results.parquet
        └── manifest.json
```

## Merge results (optional)

After both models finish, merge their outputs into an experiment-ready table:

```bash
python scripts/merge_results.py \
  --hm-conformer-results data/inference/hm-conformer/2026-06-11_hm-conformer/results.parquet \
  --quad-stream-results data/inference/quad-stream/2026-06-11_quad-stream/results.parquet \
  --samples-path data/inference/hm-conformer/2026-06-11_hm-conformer/samples.parquet \
  --output-dir data/inference/merged
```

This produces:

```text
data/inference/merged/2026-06-11_hm-conformer_quad-stream/
├── samples.parquet
├── classifier_results.parquet
└── manifest.json
```

## Validation

Each run validates its outputs automatically before finishing:

- required parquet/json files exist
- `sample_id` values are unique
- predictions are only `real` or `deepfake`
- confidence is in `[0, 1]`
- error types are only `none`, `fp`, or `fn`
- result row count matches sample count

The merge step validates that both models share the same `sample_id` space.

## Tests

Unit tests cover metadata loading, score normalization, export format, validation rules, and merge logic. They do **not** download data or load model checkpoints.

```bash
source .venv/bin/activate
pytest tests/ -v
```

## VM deployment notes

On a Google Cloud VM:

1. Clone this repository and download the dataset into `data/raw/`.
2. Copy model checkpoints into `models/hm-conformer/params/` and `models/quad-stream/checkpoints/`.
3. Precompute quad-stream features once.
4. Run hm-conformer and quad-stream inference separately.
5. Optionally merge outputs and copy `data/inference/merged/` to the experiment repository.

The experiment repository only needs the merged score files; it does not need to know how either model was executed.
