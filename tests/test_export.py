import json

import pandas as pd

from scripts.inference.export import make_run_id, write_run_outputs


def test_make_run_id_format():
    assert make_run_id("hm-conformer").endswith("_hm-conformer")


def test_write_run_outputs_creates_files(tmp_path, sample_rows, result_rows):
    output_dir = tmp_path / "run"
    manifest = {
        "run_id": "2026-06-11_hm-conformer",
        "created_at": "2026-06-11T15:30:00Z",
        "model_name": "hm-conformer",
        "dataset_name": "test_dataset",
        "num_samples": len(sample_rows),
        "checkpoint_path": "models/hm-conformer/params",
        "output_files": {},
    }

    write_run_outputs(
        output_dir=output_dir,
        samples=sample_rows,
        results=result_rows,
        manifest=manifest,
    )

    assert (output_dir / "samples.parquet").exists()
    assert (output_dir / "results.parquet").exists()
    assert (output_dir / "manifest.json").exists()

    samples = pd.read_parquet(output_dir / "samples.parquet")
    results = pd.read_parquet(output_dir / "results.parquet")
    saved_manifest = json.loads((output_dir / "manifest.json").read_text())

    assert set(samples.columns) >= {
        "sample_id",
        "filename",
        "original_path",
        "ground_truth",
        "language",
        "manipulation_type",
    }
    assert set(results.columns) >= {
        "sample_id",
        "model_name",
        "prediction",
        "confidence",
        "error_type",
    }
    assert saved_manifest["model_name"] == "hm-conformer"
