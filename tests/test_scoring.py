from scripts.inference.scoring import (
    compute_error_type,
    normalize_label,
    scores_from_hm_conformer,
    scores_from_quad_stream,
)


def test_normalize_label_maps_fake_to_deepfake():
    assert normalize_label("fake") == "deepfake"
    assert normalize_label("real") == "real"


def test_hm_conformer_scores_match_spec_example():
    scores = scores_from_hm_conformer(0.087)
    assert scores["prediction"] == "real"
    assert scores["confidence"] == 0.913
    assert scores["real_score"] == 0.913
    assert scores["deepfake_score"] == 0.087


def test_quad_stream_scores_match_spec_example():
    scores = scores_from_quad_stream(0.621)
    assert scores["prediction"] == "deepfake"
    assert scores["confidence"] == 0.621
    assert scores["real_score"] == 0.379
    assert scores["deepfake_score"] == 0.621


def test_compute_error_type():
    assert compute_error_type("real", "real") == "none"
    assert compute_error_type("real", "deepfake") == "fp"
    assert compute_error_type("deepfake", "real") == "fn"
    assert compute_error_type("deepfake", "deepfake") == "none"
