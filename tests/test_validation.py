import json

import pandas as pd
import pytest

from scripts.inference.validation import (
    validate_merge_inputs,
    validate_merged_output,
    validate_run_directory,
)


def test_validate_run_directory_passes(run_dir):
    validate_run_directory(run_dir)


def test_validate_run_directory_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_run_directory(tmp_path)


def test_validate_run_directory_invalid_prediction(run_dir):
    results = pd.read_parquet(run_dir / "results.parquet")
    results.loc[0, "prediction"] = "maybe_fake"
    results.to_parquet(run_dir / "results.parquet", index=False)

    with pytest.raises(ValueError, match="Invalid prediction"):
        validate_run_directory(run_dir)


def test_validate_run_directory_invalid_confidence(run_dir):
    results = pd.read_parquet(run_dir / "results.parquet")
    results.loc[0, "confidence"] = 1.5
    results.to_parquet(run_dir / "results.parquet", index=False)

    with pytest.raises(ValueError, match="confidence"):
        validate_run_directory(run_dir)


def test_validate_run_directory_real_manipulation_type(run_dir):
    samples = pd.read_parquet(run_dir / "samples.parquet")
    samples.loc[0, "manipulation_type"] = "tts"
    samples.to_parquet(run_dir / "samples.parquet", index=False)

    with pytest.raises(ValueError, match="manipulation_type"):
        validate_run_directory(run_dir)
