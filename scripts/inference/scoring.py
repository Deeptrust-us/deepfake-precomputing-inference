"""Score normalization and error-type helpers."""

from __future__ import annotations

from typing import Literal

Prediction = Literal["real", "deepfake"]
ErrorType = Literal["none", "fp", "fn"]
GroundTruth = Literal["real", "deepfake"]

PREDICTIONS = {"real", "deepfake"}
ERROR_TYPES = {"none", "fp", "fn"}


def normalize_label(label: str) -> GroundTruth:
    """Map dataset labels (real/fake) to spec ground_truth values."""
    normalized = label.strip().lower()
    if normalized == "real":
        return "real"
    if normalized in {"fake", "deepfake", "spoof"}:
        return "deepfake"
    raise ValueError(f"Unknown label: {label!r} (expected 'real' or 'fake')")


def manipulation_type_for_entry(ground_truth: GroundTruth, model_or_speaker: str | None) -> str:
    if ground_truth == "real":
        return "none"
    if model_or_speaker:
        return model_or_speaker
    return "unknown"


def scores_from_hm_conformer(deepfake_raw: float, threshold: float = 0.5) -> dict[str, float | Prediction]:
    """Convert HM-Conformer OCSoftmax output into spec-compliant scores."""
    deepfake_score = float(max(0.0, min(1.0, deepfake_raw)))
    real_score = 1.0 - deepfake_score
    if deepfake_score < threshold:
        prediction: Prediction = "real"
        confidence = real_score
    else:
        prediction = "deepfake"
        confidence = deepfake_score
    return {
        "prediction": prediction,
        "confidence": confidence,
        "real_score": real_score,
        "deepfake_score": deepfake_score,
    }


def scores_from_quad_stream(deepfake_prob: float, threshold: float = 0.5) -> dict[str, float | Prediction]:
    """Convert QuadStream sigmoid output into spec-compliant scores."""
    deepfake_score = float(max(0.0, min(1.0, deepfake_prob)))
    real_score = 1.0 - deepfake_score
    if deepfake_score >= threshold:
        prediction: Prediction = "deepfake"
        confidence = deepfake_score
    else:
        prediction = "real"
        confidence = real_score
    return {
        "prediction": prediction,
        "confidence": confidence,
        "real_score": real_score,
        "deepfake_score": deepfake_score,
    }


def compute_error_type(ground_truth: GroundTruth, prediction: Prediction) -> ErrorType:
    if ground_truth == prediction:
        return "none"
    if ground_truth == "real" and prediction == "deepfake":
        return "fp"
    return "fn"
