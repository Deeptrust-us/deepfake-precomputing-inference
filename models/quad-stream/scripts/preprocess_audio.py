"""Preprocess an audio deepfake dataset into 4 frequency-image streams.

Generates (1,224,224) float32 .npy files:
  - features/segment_stft/<stem>_<seg_num>.npy
  - features/segment_logmel/<stem>_<seg_num>.npy
  - features/full_stft/<stem>.npy
  - features/full_logmel/<stem>.npy
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add parent directory to path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # noqa: E402

from src.utils.audio_features import (  # noqa: E402
    AudioFeatureConfig,
    compute_logmel_image,
    compute_stft_image,
    extract_fixed_length_segment,
    get_speech_segments,
    load_audio_mono,
    save_feature_npy,
)


def _resolve_wav(dataset_root: Path, filename: str, label: str | None) -> Path:
    cand = dataset_root / "audio" / filename
    if cand.exists():
        return cand
    if label:
        cand2 = dataset_root / "audio" / label.lower() / filename
        if cand2.exists():
            return cand2
    for sub in ("real", "fake"):
        cand3 = dataset_root / "audio" / sub / filename
        if cand3.exists():
            return cand3
    return cand


def _process_one(
    dataset_root: str,
    out_root: str,
    entry: dict,
    entry_idx: int,
    cfg_dict: dict,
    segment_mode: str,
    seed: int | None,
    overwrite: bool,
    delete_audio: bool,
    delete_audio_dry_run: bool,
    torch_threads: int | None,
) -> tuple[int, int, int, int]:
    """Process one labels.json entry.

    Returns: (processed_count, skipped_count, deleted_count, delete_failed_count)
    """
    filename = entry.get("filename") or "<missing filename>"
    label = entry.get("label")
    wav_path: Path | None = None

    try:
        if torch_threads is not None and torch_threads > 0:
            torch.set_num_threads(int(torch_threads))

        dataset_root_p = Path(dataset_root)
        out_root_p = Path(out_root)
        cfg = AudioFeatureConfig(**cfg_dict)

        if filename == "<missing filename>":
            raise ValueError(f"Entry {entry_idx} missing filename")

        stem = Path(filename).stem
        full_stft_path = out_root_p / "full_stft" / f"{stem}.npy"
        full_logmel_path = out_root_p / "full_logmel" / f"{stem}.npy"

        wav_path = _resolve_wav(dataset_root_p, filename, label)
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV not found: expected under {dataset_root_p}/audio/")

        wav_cpu, _ = load_audio_mono(wav_path, target_sr=cfg.sample_rate)

        # Full-audio features (saved once per file)
        wav_full = wav_cpu
        if overwrite or (not full_stft_path.exists()):
            save_feature_npy(full_stft_path, compute_stft_image(wav_full, cfg))
        if overwrite or (not full_logmel_path.exists()):
            save_feature_npy(full_logmel_path, compute_logmel_image(wav_full, cfg))

        # Segment features (saved for each VAD segment) — VAD runs on CPU
        segments = get_speech_segments(wav_cpu, cfg.sample_rate)
        if not segments:
            segments = [(0, len(wav_cpu))]

        if segment_mode == "random_one":
            rng = random.Random(seed)
            chosen_seg_num = rng.randrange(len(segments))
            seg_iter = [(chosen_seg_num, segments[chosen_seg_num])]
        else:
            seg_iter = list(enumerate(segments))

        expected_segment_paths: list[Path] = []
        for seg_num, (s, e) in seg_iter:
            seg_cpu = extract_fixed_length_segment(wav_cpu, cfg.sample_rate, s, e, cfg.segment_seconds)
            seg = seg_cpu

            seg_stft_path = out_root_p / "segment_stft" / f"{stem}_{seg_num}.npy"
            seg_logmel_path = out_root_p / "segment_logmel" / f"{stem}_{seg_num}.npy"
            expected_segment_paths.extend([seg_stft_path, seg_logmel_path])

            if overwrite or (not seg_stft_path.exists()):
                save_feature_npy(seg_stft_path, compute_stft_image(seg, cfg))
            if overwrite or (not seg_logmel_path.exists()):
                save_feature_npy(seg_logmel_path, compute_logmel_image(seg, cfg))

        deleted = 0
        delete_failed = 0
        if delete_audio or delete_audio_dry_run:
            expected = [full_stft_path, full_logmel_path, *expected_segment_paths]
            if all(p.exists() for p in expected):
                if delete_audio_dry_run:
                    print(f"[dry-run] would delete: {wav_path}", flush=True)
                else:
                    try:
                        wav_path.unlink()
                        deleted = 1
                    except Exception as ex:  # noqa: BLE001
                        delete_failed = 1
                        print(f"WARNING: failed to delete {wav_path}: {ex}", flush=True)
            else:
                missing = [str(p) for p in expected if not p.exists()]
                print(
                    f"WARNING: not deleting {wav_path} because some expected features are missing "
                    f"({len(missing)} missing).",
                    flush=True,
                )

        return (1, 0, deleted, delete_failed)
    except Exception as ex:  # noqa: BLE001
        wav_info = f", wav={wav_path}" if wav_path is not None else ""
        print(
            f"ERROR processing filename={filename}{wav_info}: {type(ex).__name__}: {ex}",
            file=sys.stderr,
            flush=True,
        )
        return (0, 1, 0, 0)


def main() -> None:
    p = argparse.ArgumentParser(description="Preprocess WAV dataset into quad-stream audio features")
    p.add_argument("--dataset-root", type=str, required=True, help="Dataset root containing labels.json and audio/")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--segment-seconds", type=float, default=2.0)
    p.add_argument("--max-full-seconds", type=float, default=None)
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes (CPU parallelism). 0 = run in current process.",
    )
    p.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Optional torch thread count per process. Useful with --num-workers to avoid oversubscription.",
    )
    p.add_argument(
        "--segment-mode",
        type=str,
        choices=("all", "random_one"),
        default="all",
        help="How to write segment features: 'all' writes one pair per VAD segment; "
        "'random_one' writes exactly one randomly chosen segment per file.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed used when --segment-mode=random_one (for reproducibility).",
    )
    p.add_argument(
        "--delete-audio",
        action="store_true",
        help="Delete the source .wav after feature extraction (saves disk). "
        "Safety: deletion only happens if all expected output feature files exist. "
        "WARNING: you won't be able to re-preprocess or use compute_on_the_fly without the audio.",
    )
    p.add_argument(
        "--delete-audio-dry-run",
        action="store_true",
        help="Show which .wav files would be deleted by --delete-audio, but do not delete anything.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing feature .npy files")
    args = p.parse_args()

    dataset_root = Path(args.dataset_root)
    labels_path = dataset_root / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json not found: {labels_path}")

    entries = json.loads(labels_path.read_text())
    if not isinstance(entries, list):
        raise ValueError("labels.json must be a JSON list")

    cfg = AudioFeatureConfig(
        sample_rate=args.sample_rate,
        segment_seconds=float(args.segment_seconds),
        max_full_seconds=args.max_full_seconds,
    )

    out_root = dataset_root / "features"
    (out_root / "segment_stft").mkdir(parents=True, exist_ok=True)
    (out_root / "segment_logmel").mkdir(parents=True, exist_ok=True)
    (out_root / "full_stft").mkdir(parents=True, exist_ok=True)
    (out_root / "full_logmel").mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    deleted = 0
    delete_failed = 0

    cfg_dict = {
        "sample_rate": cfg.sample_rate,
        "segment_seconds": cfg.segment_seconds,
        "max_full_seconds": cfg.max_full_seconds,
        "stft_n_fft": cfg.stft_n_fft,
        "stft_hop_length": cfg.stft_hop_length,
        "stft_win_length": cfg.stft_win_length,
        "mel_n_fft": cfg.mel_n_fft,
        "mel_hop_length": cfg.mel_hop_length,
        "mel_win_length": cfg.mel_win_length,
        "mel_n_mels": cfg.mel_n_mels,
        "mel_f_min": cfg.mel_f_min,
        "mel_f_max": cfg.mel_f_max,
        "out_size": cfg.out_size,
    }

    if args.num_workers and args.num_workers > 0:
        # Multiprocessing for throughput across files.
        # Note: with multiprocessing you typically want fewer torch threads per worker.
        with ProcessPoolExecutor(max_workers=int(args.num_workers)) as ex:
            future_to_filename: dict = {}
            futures = []
            for i, e in enumerate(entries):
                fut = ex.submit(
                    _process_one,
                    str(dataset_root),
                    str(out_root),
                    e,
                    i,
                    cfg_dict,
                    args.segment_mode,
                    (args.seed + i) if args.seed is not None else None,
                    bool(args.overwrite),
                    bool(args.delete_audio),
                    bool(args.delete_audio_dry_run),
                    args.torch_threads,
                )
                futures.append(fut)
                future_to_filename[fut] = e.get("filename", "<unknown filename>")

            for fut in as_completed(futures):
                try:
                    p1, s1, d1, df1 = fut.result()
                except Exception as ex2:  # noqa: BLE001
                    skipped += 1
                    fn = future_to_filename.get(fut, "<unknown filename>")
                    print(
                        f"ERROR processing filename={fn}: {type(ex2).__name__}: {ex2}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue

                processed += p1
                skipped += s1
                deleted += d1
                delete_failed += df1
                done = processed + skipped
                if done % 50 == 0:
                    print(f"Progress: done={done} (processed={processed}, skipped={skipped})...", flush=True)
    else:
        # Single process
        for i, e in enumerate(entries):
            seed_for_entry = (args.seed + i) if args.seed is not None else None
            p1, s1, d1, df1 = _process_one(
                str(dataset_root),
                str(out_root),
                e,
                i,
                cfg_dict,
                args.segment_mode,
                seed_for_entry,
                bool(args.overwrite),
                bool(args.delete_audio),
                bool(args.delete_audio_dry_run),
                args.torch_threads,
            )
            processed += p1
            skipped += s1
            deleted += d1
            delete_failed += df1
            done = processed + skipped
            if done % 50 == 0:
                print(f"Progress: done={done} (processed={processed}, skipped={skipped})...", flush=True)

    print(
        f"Done. processed={processed}, skipped={skipped}, deleted_audio={deleted}, "
        f"delete_failed={delete_failed}, features_dir={out_root}"
    )


if __name__ == "__main__":
    main()

