"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.inference.export import build_manifest, write_run_outputs
from scripts.inference.scoring import compute_error_type, scores_from_hm_conformer, scores_from_quad_stream


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def labels_mini_path() -> Path:
    return FIXTURES_DIR / "labels_mini.json"


@pytest.fixture
def sample_rows() -> list[dict]:
    return [
        {
            "sample_id": "0000407282",
            "filename": "0000407282.wav",
            "original_path": "/tmp/dataset/real/en/sample.wav",
            "ground_truth": "real",
            "language": "en",
            "manipulation_type": "none",
            "model_or_speaker": "speaker_a",
        },
        {
            "sample_id": "0000407283",
            "filename": "0000407283.wav",
            "original_path": "/tmp/dataset/fake/en/OuteTTS/sample.wav",
            "ground_truth": "deepfake",
            "language": "en",
            "manipulation_type": "OuteTTS",
            "model_or_speaker": "OuteTTS",
        },
    ]


@pytest.fixture
def result_rows(sample_rows) -> list[dict]:
    rows = []
    for sample in sample_rows:
        scores = scores_from_hm_conformer(0.12 if sample["ground_truth"] == "real" else 0.88)
        prediction = scores["prediction"]
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "model_name": "hm-conformer",
                "prediction": prediction,
                "confidence": scores["confidence"],
                "real_score": scores["real_score"],
                "deepfake_score": scores["deepfake_score"],
                "error_type": compute_error_type(sample["ground_truth"], prediction),
            }
        )
    return rows


@pytest.fixture
def run_dir(tmp_path, sample_rows, result_rows) -> Path:
    from datetime import datetime, timezone

    output_dir = tmp_path / "hm-conformer" / "2026-06-11_hm-conformer"
    started = datetime(2026, 6, 11, 15, 0, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 6, 11, 15, 30, 0, tzinfo=timezone.utc)
    manifest = build_manifest(
        run_id="2026-06-11_hm-conformer",
        model_name="hm-conformer",
        dataset_name="test_dataset",
        num_samples=len(sample_rows),
        checkpoint_path="models/hm-conformer/params",
        output_dir=output_dir,
        started_at=started,
        finished_at=finished,
    )
    write_run_outputs(output_dir=output_dir, samples=sample_rows, results=result_rows, manifest=manifest)
    return output_dir
