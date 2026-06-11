from pathlib import Path

from scripts.inference.dataset import build_samples_table, entry_to_sample, resolve_audio_path


def test_entry_to_sample_real(labels_mini_path):
    entries = __import__("json").loads(labels_mini_path.read_text())
    sample = entry_to_sample(entries[0])

    assert sample["sample_id"] == "0000407282"
    assert sample["ground_truth"] == "real"
    assert sample["manipulation_type"] == "none"
    assert sample["language"] == "en"


def test_entry_to_sample_deepfake(labels_mini_path):
    entries = __import__("json").loads(labels_mini_path.read_text())
    sample = entry_to_sample(entries[1])

    assert sample["ground_truth"] == "deepfake"
    assert sample["manipulation_type"] == "OuteTTS"


def test_build_samples_table_unique_ids(labels_mini_path):
    samples = build_samples_table(labels_mini_path)
    assert len(samples) == 3
    assert len({s["sample_id"] for s in samples}) == 3


def test_build_samples_table_optional_split(labels_mini_path):
    samples = build_samples_table(labels_mini_path)
    assert samples[2]["split"] == "test"


def test_resolve_audio_path_prefers_dataset_root(tmp_path, labels_mini_path):
    entries = __import__("json").loads(labels_mini_path.read_text())
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    target = audio_dir / entries[0]["filename"]
    target.write_bytes(b"RIFF")

    resolved = resolve_audio_path(entries[0], tmp_path)
    assert resolved == target
