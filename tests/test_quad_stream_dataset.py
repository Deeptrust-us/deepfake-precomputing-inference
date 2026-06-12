import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
QUAD_STREAM_DIR = REPO_ROOT / "models" / "quad-stream"
if str(QUAD_STREAM_DIR) not in sys.path:
    sys.path.insert(0, str(QUAD_STREAM_DIR))

from src.data.dataset import (  # noqa: E402
    DeepfakeDataset,
    SegmentFeatureIndex,
    build_segment_feature_index,
    segment_stem_from_filename,
)


def _write_feature(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.zeros((1, 224, 224), dtype=np.float32))


def test_segment_stem_from_filename_parses_segment_suffix():
    assert segment_stem_from_filename("000001_0.npy") == "000001"
    assert segment_stem_from_filename("000001.npy") is None
    assert segment_stem_from_filename("000001_x.npy") is None


def test_build_segment_feature_index_resolves_unique_stems(tmp_path: Path):
    seg_dir = tmp_path / "segment_stft"
    target = seg_dir / "000001_0.npy"
    _write_feature(target)

    index = build_segment_feature_index(seg_dir)
    assert index.resolve("000001") == target


def test_segment_feature_index_cache_reuses_build(tmp_path: Path):
    seg_dir = tmp_path / "segment_stft"
    _write_feature(seg_dir / "000001_0.npy")

    first = SegmentFeatureIndex.load_or_build(seg_dir)
    second = SegmentFeatureIndex.load_or_build(seg_dir)
    assert first.resolve("000001") == second.resolve("000001")
    assert (seg_dir / ".segment_feature_index.json").exists()


def test_build_segment_feature_index_tracks_ambiguous_stems(tmp_path: Path):
    seg_dir = tmp_path / "segment_logmel"
    _write_feature(seg_dir / "000001_0.npy")
    _write_feature(seg_dir / "000001_1.npy")

    index = build_segment_feature_index(seg_dir)
    with pytest.raises(FileNotFoundError, match="found 2"):
        index.resolve("000001")


def test_deepfake_dataset_uses_filtered_entries(tmp_path: Path):
    features_dir = tmp_path / "features"
    stem = "000001"
    _write_feature(features_dir / "segment_stft" / f"{stem}_0.npy")
    _write_feature(features_dir / "segment_logmel" / f"{stem}_0.npy")
    _write_feature(features_dir / "full_stft" / f"{stem}.npy")
    _write_feature(features_dir / "full_logmel" / f"{stem}.npy")

    entries = [
        {"filename": f"{stem}.wav", "label": "real", "id": stem},
        {"filename": "000002.wav", "label": "fake", "id": "000002"},
    ]
    dataset = DeepfakeDataset(
        dataset_root=tmp_path,
        features_dir=features_dir,
        entries=entries,
        skip_missing=True,
    )

    assert len(dataset) == 1
    item = dataset[0]
    assert item["sample_id"] == stem
