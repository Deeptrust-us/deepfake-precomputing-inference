#!/usr/bin/env python3
"""Run batch inference for hm-conformer or quad-stream over labels.json metadata."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.dataset import build_samples_table
from scripts.inference.export import build_manifest, make_run_id, write_run_outputs
from scripts.inference.runners.hm_conformer import HmConformerRunner
from scripts.inference.runners.quad_stream import QuadStreamRunner
from scripts.inference.validation import validate_run_directory


SUPPORTED_MODELS = {"hm-conformer", "quad-stream"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute deepfake classifier outputs.")
    parser.add_argument(
        "--model-name",
        required=True,
        choices=sorted(SUPPORTED_MODELS),
        help="Classifier to run.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw"),
        help="Root directory containing audio files or precomputed features.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/labels.json"),
        help="Path to labels.json metadata file.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Path to model checkpoint directory or checkpoint file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Base output directory, e.g. data/inference/hm-conformer.",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=None,
        help="Precomputed features directory for quad-stream (default: <dataset-root>/features).",
    )
    parser.add_argument(
        "--dataset-name",
        default="deepfake_audio_dataset_v1",
        help="Dataset name stored in manifest.json.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier. Defaults to YYYY-MM-DD_<model-name>.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of samples (useful for smoke tests).",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip samples with missing audio/features instead of failing.",
    )
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Load quad-stream checkpoint with weights_only=False.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    run_id = args.run_id or make_run_id(args.model_name, started_at)
    run_dir = args.output_dir / run_id

    metadata_path = args.metadata_path
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    samples = build_samples_table(metadata_path, args.dataset_root, limit=args.limit)

    if args.model_name == "hm-conformer":
        runner = HmConformerRunner(args.checkpoint_path)
        results = runner.run(samples, skip_missing=args.skip_missing)
    else:
        runner = QuadStreamRunner(
            args.checkpoint_path,
            features_dir=args.features_dir,
            trust_checkpoint=args.trust_checkpoint,
        )
        results = runner.run(
            samples,
            dataset_root=args.dataset_root,
            labels_file=metadata_path,
            skip_missing=args.skip_missing,
        )

    if args.skip_missing:
        processed_ids = {r["sample_id"] for r in results}
        samples = [s for s in samples if s["sample_id"] in processed_ids]

    finished_at = datetime.now(timezone.utc)
    manifest = build_manifest(
        run_id=run_id,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        num_samples=len(samples),
        checkpoint_path=str(args.checkpoint_path),
        output_dir=run_dir,
        started_at=started_at,
        finished_at=finished_at,
    )

    export_samples = [{k: v for k, v in sample.items() if k != "resolved_audio_path"} for sample in samples]
    write_run_outputs(
        output_dir=run_dir,
        samples=export_samples,
        results=results,
        manifest=manifest,
    )
    validate_run_directory(run_dir)

    print(f"Inference complete: {run_dir}")
    print(f"  samples: {run_dir / 'samples.parquet'}")
    print(f"  results: {run_dir / 'results.parquet'}")
    print(f"  manifest: {run_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
