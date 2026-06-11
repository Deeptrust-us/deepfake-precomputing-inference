from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.inference.export import build_manifest, write_run_outputs
from scripts.inference.scoring import compute_error_type, scores_from_hm_conformer, scores_from_quad_stream
from scripts.merge_results import build_merged_table
from scripts.inference.validation import validate_merge_inputs, validate_merged_output


def _write_model_results(path: Path, model_name: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_merge_results(tmp_path, sample_rows):
    hm_rows = []
    quad_rows = []
    for sample in sample_rows:
        hm_scores = scores_from_hm_conformer(0.1)
        quad_scores = scores_from_quad_stream(0.9)
        hm_pred = hm_scores["prediction"]
        quad_pred = quad_scores["prediction"]
        hm_rows.append(
            {
                "sample_id": sample["sample_id"],
                "model_name": "hm-conformer",
                "prediction": hm_pred,
                "confidence": hm_scores["confidence"],
                "real_score": hm_scores["real_score"],
                "deepfake_score": hm_scores["deepfake_score"],
                "error_type": compute_error_type(sample["ground_truth"], hm_pred),
                "runtime_ms": 10.0,
            }
        )
        quad_rows.append(
            {
                "sample_id": sample["sample_id"],
                "model_name": "quad-stream",
                "prediction": quad_pred,
                "confidence": quad_scores["confidence"],
                "real_score": quad_scores["real_score"],
                "deepfake_score": quad_scores["deepfake_score"],
                "error_type": compute_error_type(sample["ground_truth"], quad_pred),
                "runtime_ms": 12.0,
            }
        )

    hm_path = tmp_path / "hm.parquet"
    quad_path = tmp_path / "quad.parquet"
    samples_path = tmp_path / "samples.parquet"
    _write_model_results(hm_path, "hm-conformer", hm_rows)
    _write_model_results(quad_path, "quad-stream", quad_rows)
    pd.DataFrame(sample_rows).to_parquet(samples_path, index=False)

    hm_df, quad_df, samples_df = validate_merge_inputs(hm_path, quad_path, samples_path)
    merged = build_merged_table(hm_df, quad_df)
    validate_merged_output(merged, samples_df)

    assert set(merged.columns) >= {
        "sample_id",
        "hm_conformer_prediction",
        "hm_conformer_confidence",
        "quad_stream_prediction",
        "quad_stream_confidence",
        "hm_conformer_error_type",
        "quad_stream_error_type",
    }
    assert len(merged) == len(sample_rows)
