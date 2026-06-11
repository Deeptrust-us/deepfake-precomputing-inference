#!/usr/bin/env python3
"""Merge hm-conformer and quad-stream inference outputs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.export import make_run_id
from scripts.inference.validation import validate_merge_inputs, validate_merged_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge classifier inference outputs.")
    parser.add_argument("--hm-conformer-results", type=Path, required=True)
    parser.add_argument("--quad-stream-results", type=Path, required=True)
    parser.add_argument("--samples-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def build_merged_table(hm_results: pd.DataFrame, quad_results: pd.DataFrame) -> pd.DataFrame:
    hm = hm_results.set_index("sample_id")
    quad = quad_results.set_index("sample_id")

    merged = pd.DataFrame(index=hm.index.sort_values())
    merged.index.name = "sample_id"
    merged = merged.reset_index()

    merged["hm_conformer_prediction"] = hm.loc[merged["sample_id"], "prediction"].values
    merged["hm_conformer_confidence"] = hm.loc[merged["sample_id"], "confidence"].values
    merged["quad_stream_prediction"] = quad.loc[merged["sample_id"], "prediction"].values
    merged["quad_stream_confidence"] = quad.loc[merged["sample_id"], "confidence"].values
    merged["hm_conformer_error_type"] = hm.loc[merged["sample_id"], "error_type"].values
    merged["quad_stream_error_type"] = quad.loc[merged["sample_id"], "error_type"].values

    optional_map = {
        "real_score": ("hm_conformer_real_score", "quad_stream_real_score"),
        "deepfake_score": ("hm_conformer_deepfake_score", "quad_stream_deepfake_score"),
        "runtime_ms": ("hm_conformer_runtime_ms", "quad_stream_runtime_ms"),
    }
    for source_col, (hm_col, quad_col) in optional_map.items():
        if source_col in hm.columns:
            merged[hm_col] = hm.loc[merged["sample_id"], source_col].values
        if source_col in quad.columns:
            merged[quad_col] = quad.loc[merged["sample_id"], source_col].values

    return merged


def main() -> int:
    args = parse_args()
    hm_results, quad_results, samples = validate_merge_inputs(
        args.hm_conformer_results,
        args.quad_stream_results,
        args.samples_path,
    )

    merged = build_merged_table(hm_results, quad_results)
    validate_merged_output(merged, samples)

    run_id = args.run_id or make_run_id("hm-conformer_quad-stream")
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    samples.to_parquet(output_dir / "samples.parquet", index=False)
    merged.to_parquet(output_dir / "classifier_results.parquet", index=False)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "num_samples": len(samples),
        "output_files": {
            "samples": str(output_dir / "samples.parquet"),
            "classifier_results": str(output_dir / "classifier_results.parquet"),
        },
        "inputs": {
            "hm_conformer_results": str(args.hm_conformer_results),
            "quad_stream_results": str(args.quad_stream_results),
            "samples": str(args.samples_path),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Merged output written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
