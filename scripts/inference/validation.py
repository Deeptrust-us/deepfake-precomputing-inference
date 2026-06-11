"""Validation rules for inference outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .scoring import ERROR_TYPES, PREDICTIONS


def validate_run_directory(run_dir: Path) -> None:
    samples_path = run_dir / "samples.parquet"
    results_path = run_dir / "results.parquet"
    manifest_path = run_dir / "manifest.json"

    for path in (samples_path, results_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required output file: {path}")

    samples = pd.read_parquet(samples_path)
    results = pd.read_parquet(results_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    _validate_samples(samples)
    _validate_results(results, samples)
    _validate_manifest(manifest, run_dir)


def _validate_samples(samples: pd.DataFrame) -> None:
    required = {
        "sample_id",
        "filename",
        "original_path",
        "ground_truth",
        "language",
        "manipulation_type",
    }
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"samples.parquet missing columns: {sorted(missing)}")

    if samples["sample_id"].duplicated().any():
        raise ValueError("sample_id values must be unique in samples.parquet")

    allowed_gt = {"real", "deepfake"}
    invalid_gt = set(samples["ground_truth"].unique()) - allowed_gt
    if invalid_gt:
        raise ValueError(f"Invalid ground_truth values: {sorted(invalid_gt)}")

    real_rows = samples[samples["ground_truth"] == "real"]
    if not (real_rows["manipulation_type"] == "none").all():
        raise ValueError("real samples must use manipulation_type = none")


def _validate_results(results: pd.DataFrame, samples: pd.DataFrame) -> None:
    required = {"sample_id", "model_name", "prediction", "confidence", "error_type"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results.parquet missing columns: {sorted(missing)}")

    if len(results) != len(samples):
        raise ValueError(
            f"Result row count ({len(results)}) does not match sample count ({len(samples)})"
        )

    sample_ids = set(samples["sample_id"].astype(str))
    result_ids = results["sample_id"].astype(str)
    if result_ids.duplicated().any():
        raise ValueError("sample_id values must be unique in results.parquet")

    unknown_ids = set(result_ids) - sample_ids
    if unknown_ids:
        raise ValueError(f"results contain sample_id values missing from samples: {sorted(unknown_ids)[:5]}")

    invalid_predictions = set(results["prediction"].unique()) - PREDICTIONS
    if invalid_predictions:
        raise ValueError(f"Invalid prediction values: {sorted(invalid_predictions)}")

    if ((results["confidence"] < 0) | (results["confidence"] > 1)).any():
        raise ValueError("confidence values must be between 0 and 1")

    invalid_errors = set(results["error_type"].unique()) - ERROR_TYPES
    if invalid_errors:
        raise ValueError(f"Invalid error_type values: {sorted(invalid_errors)}")


def _validate_manifest(manifest: dict, run_dir: Path) -> None:
    required = {"run_id", "created_at", "model_name", "dataset_name", "num_samples", "checkpoint_path", "output_files"}
    missing = required - set(manifest.keys())
    if missing:
        raise ValueError(f"manifest.json missing fields: {sorted(missing)}")

    output_files = manifest["output_files"]
    for key in ("samples", "results"):
        if key not in output_files:
            raise ValueError(f"manifest output_files missing {key!r}")


def validate_merge_inputs(
    hm_results_path: Path,
    quad_results_path: Path,
    samples_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (hm_results_path, quad_results_path, samples_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing merge input: {path}")

    hm_results = pd.read_parquet(hm_results_path)
    quad_results = pd.read_parquet(quad_results_path)
    samples = pd.read_parquet(samples_path)

    if hm_results["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values in hm-conformer results")
    if quad_results["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values in quad-stream results")

    hm_ids = set(hm_results["sample_id"].astype(str))
    quad_ids = set(quad_results["sample_id"].astype(str))
    if hm_ids != quad_ids:
        only_hm = sorted(hm_ids - quad_ids)[:5]
        only_quad = sorted(quad_ids - hm_ids)[:5]
        raise ValueError(
            "hm-conformer and quad-stream results must share the same sample_id space. "
            f"Only in hm-conformer: {only_hm}; only in quad-stream: {only_quad}"
        )

    sample_ids = set(samples["sample_id"].astype(str))
    if hm_ids - sample_ids:
        raise ValueError("Merged sample_id values must exist in samples.parquet")

    return hm_results, quad_results, samples


def validate_merged_output(merged: pd.DataFrame, samples: pd.DataFrame) -> None:
    required = {
        "sample_id",
        "hm_conformer_prediction",
        "hm_conformer_confidence",
        "quad_stream_prediction",
        "quad_stream_confidence",
        "hm_conformer_error_type",
        "quad_stream_error_type",
    }
    missing = required - set(merged.columns)
    if missing:
        raise ValueError(f"classifier_results.parquet missing columns: {sorted(missing)}")

    if len(merged) != len(samples):
        raise ValueError("Merged output must have one row per sample")

    if merged["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values in merged output")
