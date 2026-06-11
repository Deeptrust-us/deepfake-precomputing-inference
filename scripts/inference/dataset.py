"""Dataset metadata loading from labels.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .scoring import manipulation_type_for_entry, normalize_label


def load_labels(metadata_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{metadata_path} must contain a JSON list of sample entries")
    return raw


def entry_to_sample(entry: dict[str, Any], dataset_root: Path | None = None) -> dict[str, Any]:
    """Map a labels.json entry to the samples.parquet schema."""
    sample_id = str(entry.get("id") or entry.get("sample_id") or Path(entry["filename"]).stem)
    filename = entry["filename"]
    ground_truth = normalize_label(str(entry["label"]))
    model_or_speaker = entry.get("model_or_speaker")

    sample: dict[str, Any] = {
        "sample_id": sample_id,
        "filename": filename,
        "original_path": entry.get("original_path", ""),
        "ground_truth": ground_truth,
        "language": entry.get("language", "unknown"),
        "manipulation_type": manipulation_type_for_entry(ground_truth, model_or_speaker),
    }

    for optional in ("split", "quality", "duration_seconds", "source_dataset", "model_or_speaker"):
        if optional in entry and entry[optional] is not None:
            sample[optional] = entry[optional]

    if dataset_root is not None:
        sample["resolved_audio_path"] = str(resolve_audio_path(entry, dataset_root))

    return sample


def build_samples_table(
    metadata_path: Path,
    dataset_root: Path | None = None,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    entries = load_labels(metadata_path)
    if limit is not None:
        entries = entries[:limit]

    samples = [entry_to_sample(entry, dataset_root) for entry in entries]
    sample_ids = [s["sample_id"] for s in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be unique in the metadata file")

    return samples


def resolve_audio_path(entry: dict[str, Any], dataset_root: Path) -> Path:
    """Resolve the on-disk audio path for HM-Conformer inference."""
    filename = entry["filename"]
    candidates = [
        dataset_root / filename,
        dataset_root / "audio" / filename,
    ]
    original_path = entry.get("original_path")
    if original_path:
        candidates.append(Path(original_path))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return dataset_root / filename
