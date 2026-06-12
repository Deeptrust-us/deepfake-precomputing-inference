"""Audio deepfake dataset loader (4-stream frequency-image inputs)."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
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


def segment_stem_from_filename(filename: str) -> str | None:
    """Parse `{stem}_{seg_num}.npy` segment feature filenames."""
    if not filename.endswith(".npy"):
        return None
    base = filename[: -len(".npy")]
    if "_" not in base:
        return None
    stem, seg_part = base.rsplit("_", 1)
    if not stem or not seg_part.isdigit():
        return None
    return stem


@dataclass(frozen=True)
class SegmentFeatureIndex:
    paths: dict[str, Path]
    ambiguous: dict[str, list[Path]]

    def resolve(self, stem: str) -> Path:
        if stem in self.ambiguous:
            found = len(self.ambiguous[stem])
            raise FileNotFoundError(
                f"Expected exactly one segment feature for stem={stem!r}; found {found}."
            )
        path = self.paths.get(stem)
        if path is None:
            raise FileNotFoundError(f"Missing segment feature for stem={stem!r}.")
        return path


def build_segment_feature_index(directory: Path, *, debug: bool = False) -> SegmentFeatureIndex:
    """Scan a segment feature directory once and build stem -> path lookups."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    if directory.exists():
        started = time.perf_counter()
        with os.scandir(directory) as iterator:
            for entry in iterator:
                if not entry.is_file():
                    continue
                stem = segment_stem_from_filename(entry.name)
                if stem is None:
                    continue
                grouped[stem].append(Path(entry.path))
        if debug:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            print(
                f"[debug][dataset] indexed {directory.name}: "
                f"files={sum(len(paths) for paths in grouped.values())} "
                f"stems={len(grouped)} elapsed_ms={elapsed_ms:.1f}",
                file=sys.stderr,
                flush=True,
            )

    paths = {stem: matches[0] for stem, matches in grouped.items() if len(matches) == 1}
    ambiguous = {stem: matches for stem, matches in grouped.items() if len(matches) > 1}
    return SegmentFeatureIndex(paths=paths, ambiguous=ambiguous)


class DeepfakeDataset(Dataset):
    """Dataset loader: 1 audio file -> 4 features.

    Each labels entry must contain at least:
      - filename: str
      - label: "real" or "fake" (case-insensitive)

    For each entry, this loads exactly 4 precomputed numpy arrays from disk (features/...):
      - features/segment_stft/<stem>_<seg_num>.npy
      - features/segment_logmel/<stem>_<seg_num>.npy
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
        *,
        entries: Optional[List[Dict[str, Any]]] = None,
        skip_missing: bool = False,
        debug: bool = False,
    ):
        self.dataset_root = Path(dataset_root)
        self.labels_file = Path(labels_file) if labels_file is not None else self.dataset_root / "labels.json"
        self.features_dir = Path(features_dir) if features_dir is not None else self.dataset_root / "features"
        self.debug = debug

        if entries is not None:
            self.entries = self._validate_entries(entries)
        else:
            if not self.labels_file.exists():
                raise FileNotFoundError(f"labels.json not found: {self.labels_file}")
            raw = json.loads(self.labels_file.read_text())
            if not isinstance(raw, list):
                raise ValueError(f"{self.labels_file} must be a JSON list of entries")
            self.entries = self._validate_entries(raw)

        self._segment_stft_index = build_segment_feature_index(
            self.features_dir / "segment_stft",
            debug=debug,
        )
        self._segment_logmel_index = build_segment_feature_index(
            self.features_dir / "segment_logmel",
            debug=debug,
        )

        if skip_missing:
            ready: list[dict[str, Any]] = []
            for entry in self.entries:
                stem = Path(entry["filename"]).stem
                try:
                    self._resolve_feature_paths(stem)
                    ready.append(entry)
                except FileNotFoundError:
                    continue
            self.entries = ready

    @staticmethod
    def _validate_entries(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for idx, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ValueError(f"labels entry at index {idx} must be a JSON object")
            filename = entry.get("filename")
            label_str = entry.get("label")
            if not filename or not label_str:
                raise ValueError(f"labels entry at index {idx} missing 'filename' or 'label'")
            _ = _label_to_int(str(label_str))
            entries.append(entry)
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def _debug(self, message: str) -> None:
        if self.debug:
            print(f"[debug][dataset] {message}", file=sys.stderr, flush=True)

    def _feature_path(self, kind: str, stem: str) -> Path:
        return self._resolve_feature_paths(stem)[kind]

    def _resolve_feature_paths(self, stem: str) -> dict[str, Path]:
        paths = {
            "segment_stft": self._segment_stft_index.resolve(stem),
            "segment_logmel": self._segment_logmel_index.resolve(stem),
            "full_stft": self.features_dir / "full_stft" / f"{stem}.npy",
            "full_logmel": self.features_dir / "full_logmel" / f"{stem}.npy",
        }
        missing = [kind for kind, path in paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing precomputed feature(s) {missing} for stem={stem!r}. "
                f"Expected under {self.features_dir}."
            )
        return paths

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item_started = time.perf_counter()
        entry = self.entries[idx]
        filename = entry.get("filename")
        if not filename:
            raise ValueError(f"labels.json entry missing 'filename' at index {idx}")

        label_str = entry.get("label")
        if not label_str:
            raise ValueError(f"labels.json entry missing 'label' at index {idx}")

        y = _label_to_int(label_str)
        stem = Path(filename).stem
        if self.debug:
            self._debug(f"getitem start idx={idx} stem={stem}")

        paths = self._resolve_feature_paths(stem)

        feats: Dict[str, np.ndarray] = {}
        for kind, path in paths.items():
            load_started = time.perf_counter()
            feats[kind] = np.load(str(path))
            if self.debug:
                load_ms = (time.perf_counter() - load_started) * 1000.0
                self._debug(f"np.load kind={kind} path={path.name} elapsed_ms={load_ms:.1f}")

        tensor_started = time.perf_counter()
        tensors = {k: torch.from_numpy(v).to(torch.float32) for k, v in feats.items()}
        if self.debug:
            tensor_ms = (time.perf_counter() - tensor_started) * 1000.0
            total_ms = (time.perf_counter() - item_started) * 1000.0
            self._debug(f"tensor convert elapsed_ms={tensor_ms:.1f} getitem total_ms={total_ms:.1f}")

        for k, t in tensors.items():
            if t.ndim != 3 or t.shape[0] != 1:
                raise ValueError(f"Feature {k} has shape {tuple(t.shape)} (expected (1,224,224))")

        return {
            "segment_stft": tensors["segment_stft"],
            "segment_logmel": tensors["segment_logmel"],
            "full_stft": tensors["full_stft"],
            "full_logmel": tensors["full_logmel"],
            "label": torch.tensor(y, dtype=torch.long),
            "sample_id": stem,
            "filename": filename,
        }
