"""Audio deepfake dataset loader (4-stream frequency-image inputs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

def _label_to_int(label: str) -> int:
    label_l = label.strip().lower()
    if label_l == "real":
        return 0
    if label_l == "fake":
        return 1
    raise ValueError(f"Unknown label: {label!r} (expected 'real' or 'fake')")


class DeepfakeDataset(Dataset):
    """Dataset loader: 1 audio file -> 4 features.

    Each labels entry must contain at least:
      - filename: str
      - label: "real" or "fake" (case-insensitive)

    For each entry, this loads exactly 4 precomputed numpy arrays from disk (features/...):
      - features/segment_stft/<stem>.npy
      - features/segment_logmel/<stem>.npy
      - features/full_stft/<stem>.npy
      - features/full_logmel/<stem>.npy

    Each array must have shape (1, 224, 224) and will be converted to float32 torch tensors.
    Missing files raise FileNotFoundError.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        labels_file: Optional[str | Path] = None,
        features_dir: Optional[str | Path] = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.labels_file = Path(labels_file) if labels_file is not None else self.dataset_root / "labels.json"
        self.features_dir = Path(features_dir) if features_dir is not None else self.dataset_root / "features"

        if not self.labels_file.exists():
            raise FileNotFoundError(f"labels.json not found: {self.labels_file}")

        raw = json.loads(self.labels_file.read_text())
        if not isinstance(raw, list):
            raise ValueError(f"{self.labels_file} must be a JSON list of entries")

        # Deterministic: one dataset item per labels entry (no segmentation/VAD expansion)
        self.entries: List[Dict[str, Any]] = []
        for idx, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ValueError(f"labels entry at index {idx} must be a JSON object")
            filename = entry.get("filename")
            label_str = entry.get("label")
            if not filename or not label_str:
                raise ValueError(f"labels entry at index {idx} missing 'filename' or 'label'")
            # Validate label early for deterministic failures
            _ = _label_to_int(str(label_str))
            self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def _feature_path(self, kind: str, stem: str) -> Path:
        if kind in ("segment_stft", "segment_logmel"):
            seg_dir = self.features_dir / kind
            matches = sorted(seg_dir.glob(f"{stem}_*.npy"))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one segment feature for kind={kind!r}, stem={stem!r} "
                    f"matching pattern {seg_dir / (stem + '_*.npy')}; found {len(matches)}."
                )
            return matches[0]
        if kind in ("full_stft", "full_logmel"):
            return self.features_dir / kind / f"{stem}.npy"
        raise ValueError(f"Unknown feature kind: {kind!r}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        filename = entry.get("filename")
        if not filename:
            raise ValueError(f"labels.json entry missing 'filename' at index {idx}")

        label_str = entry.get("label")
        if not label_str:
            raise ValueError(f"labels.json entry missing 'label' at index {idx}")

        y = _label_to_int(label_str)
        stem = Path(filename).stem

        paths = {
            "segment_stft": self._feature_path("segment_stft", stem),
            "segment_logmel": self._feature_path("segment_logmel", stem),
            "full_stft": self._feature_path("full_stft", stem),
            "full_logmel": self._feature_path("full_logmel", stem),
        }

        missing = [k for k, p in paths.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing precomputed feature(s) {missing} for stem={stem!r}. "
                f"Expected under {self.features_dir}."
            )

        feats = {k: np.load(str(p)) for k, p in paths.items()}
        tensors = {k: torch.from_numpy(v).to(torch.float32) for k, v in feats.items()}

        # Ensure shapes are (1, 224, 224)
        for k, t in tensors.items():
            if t.ndim != 3 or t.shape[0] != 1:
                raise ValueError(f"Feature {k} has shape {tuple(t.shape)} (expected (1,224,224))")

        return {
            "segment_stft": tensors["segment_stft"],
            "segment_logmel": tensors["segment_logmel"],
            "full_stft": tensors["full_stft"],
            "full_logmel": tensors["full_logmel"],
            "label": torch.tensor(y, dtype=torch.long),
            # Optional lightweight metadata for logging/debugging:
            "sample_id": stem,
            "filename": filename,
        }

