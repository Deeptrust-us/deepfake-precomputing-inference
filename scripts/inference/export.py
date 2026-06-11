"""Export inference outputs to parquet and manifest.json."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def make_run_id(model_name: str, when: datetime | None = None) -> str:
    when = when or datetime.now(timezone.utc)
    return f"{when.strftime('%Y-%m-%d')}_{model_name}"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _gpu_name() -> str | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


def build_manifest(
    *,
    run_id: str,
    model_name: str,
    dataset_name: str,
    num_samples: int,
    checkpoint_path: str,
    output_dir: Path,
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, Any]:
    samples_path = output_dir / "samples.parquet"
    results_path = output_dir / "results.parquet"
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": finished_at.isoformat().replace("+00:00", "Z"),
        "model_name": model_name,
        "dataset_name": dataset_name,
        "num_samples": num_samples,
        "checkpoint_path": checkpoint_path,
        "output_files": {
            "samples": str(samples_path),
            "results": str(results_path),
        },
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "finished_at": finished_at.isoformat().replace("+00:00", "Z"),
    }
    gpu_name = _gpu_name()
    if gpu_name:
        manifest["gpu_name"] = gpu_name
    return manifest


def write_run_outputs(
    *,
    output_dir: Path,
    samples: list[dict[str, Any]],
    results: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_df = pd.DataFrame(samples)
    results_df = pd.DataFrame(results)

    export_columns = [
        "sample_id",
        "filename",
        "original_path",
        "ground_truth",
        "language",
        "manipulation_type",
    ]
    optional_sample_columns = ["split", "quality", "duration_seconds", "source_dataset", "model_or_speaker"]
    for column in optional_sample_columns:
        if column in samples_df.columns:
            export_columns.append(column)
    samples_df = samples_df[[c for c in export_columns if c in samples_df.columns]]

    results_df.to_parquet(output_dir / "results.parquet", index=False)
    samples_df.to_parquet(output_dir / "samples.parquet", index=False)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
