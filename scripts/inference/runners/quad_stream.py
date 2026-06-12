"""QuadStream batch inference runner."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..scoring import compute_error_type, scores_from_quad_stream


def _debug_log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[debug][quad-stream] {message}", file=sys.stderr, flush=True)


def _configure_cpu_threads(num_threads: int | None, *, debug: bool = False) -> int:
    cpu_count = os.cpu_count() or 1
    threads = num_threads if num_threads is not None else min(4, cpu_count)
    threads = max(1, threads)

    torch.set_num_threads(threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(max(1, min(2, threads // 2)))

    for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_name] = str(threads)

    _debug_log(debug, f"configured cpu threads={threads} (available={cpu_count})")
    return threads


def _default_num_workers(cpu_threads: int) -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 1:
        return 0
    return max(1, min(4, cpu_count - max(1, cpu_threads // 2)))


def _entries_from_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for sample in samples:
        label = sample["ground_truth"]
        if label == "deepfake":
            label = "fake"
        entries.append(
            {
                "filename": sample["filename"],
                "label": label,
                "id": sample["sample_id"],
            }
        )
    return entries


class QuadStreamRunner:
    def __init__(
        self,
        checkpoint_path: Path,
        *,
        config_path: Path | None = None,
        features_dir: Path | None = None,
        trust_checkpoint: bool = False,
    ):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.features_dir = features_dir
        self.trust_checkpoint = trust_checkpoint
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _repo_paths(self) -> tuple[Path, Path]:
        repo_root = Path(__file__).resolve().parents[3]
        quad_stream_dir = repo_root / "models" / "quad-stream"
        config_path = self.config_path or (quad_stream_dir / "config" / "config.yaml")
        return quad_stream_dir, config_path

    def _load_model(self, *, debug: bool = False):
        if self._model is not None:
            return self._model

        quad_stream_dir, config_path = self._repo_paths()
        if str(quad_stream_dir) not in sys.path:
            sys.path.insert(0, str(quad_stream_dir))

        from src.models.quad_stream import QuadStreamModel
        from src.utils.checkpoint import load_checkpoint, remap_state_dict_for_compat

        _debug_log(debug, f"loading config from {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        _debug_log(debug, f"building model on device={self._device}")
        model_cfg = config.get("model", {})
        model = QuadStreamModel(
            backbone=model_cfg.get("backbone", "resnet18"),
            feature_dim=model_cfg.get("feature_dim", 256),
            fusion_dim=model_cfg.get("fusion_dim", 512),
            dropout=model_cfg.get("dropout", 0.5),
            pretrained=model_cfg.get("pretrained", True),
            use_attention=model_cfg.get("use_attention", False),
        ).to(self._device)

        _debug_log(debug, f"loading checkpoint from {self.checkpoint_path}")
        checkpoint_started = time.perf_counter()
        checkpoint = load_checkpoint(
            str(self.checkpoint_path),
            self._device,
            trust_checkpoint=self.trust_checkpoint,
        )
        _debug_log(
            debug,
            f"checkpoint loaded elapsed_ms={(time.perf_counter() - checkpoint_started) * 1000.0:.1f}",
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        try:
            model.load_state_dict(state_dict, strict=True)
            _debug_log(debug, "checkpoint state_dict loaded (strict=True)")
        except RuntimeError:
            remapped, _ = remap_state_dict_for_compat(state_dict)
            model.load_state_dict(remapped, strict=False)
            _debug_log(debug, "checkpoint state_dict loaded (strict=False, remapped)")

        model.eval()
        self._model = model
        self._config = config
        return self._model

    def dry_run(
        self,
        samples: list[dict[str, Any]],
        *,
        dataset_root: Path,
        labels_file: Path,
        skip_missing: bool = False,
        debug: bool = False,
    ) -> dict[str, Any]:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {self.checkpoint_path}")

        quad_stream_dir, config_path = self._repo_paths()
        if not config_path.exists():
            raise FileNotFoundError(f"Quad-stream config not found: {config_path}")

        if str(quad_stream_dir) not in sys.path:
            sys.path.insert(0, str(quad_stream_dir))

        from src.data.dataset import DeepfakeDataset

        features_dir = self.features_dir or (dataset_root / "features")
        if not features_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {features_dir}")

        _debug_log(debug, "dry-run: loading model")
        self._load_model(debug=debug)
        entries = _entries_from_samples(samples)
        _debug_log(debug, f"dry-run: building dataset features_dir={features_dir} entries={len(entries)}")
        dataset = DeepfakeDataset(
            dataset_root=dataset_root,
            features_dir=features_dir,
            entries=entries,
            skip_missing=skip_missing,
            debug=debug,
        )

        ready_ids = [Path(entry["filename"]).stem for entry in dataset.entries]
        ready_set = set(ready_ids)
        missing = [
            {"sample_id": sample["sample_id"], "error": "missing precomputed features"}
            for sample in samples
            if sample["sample_id"] not in ready_set
        ]

        if missing and not skip_missing:
            first = missing[0]
            raise FileNotFoundError(
                f"Missing precomputed features for {first['sample_id']}: {first['error']} "
                f"({len(missing)} missing of {len(samples)} samples)"
            )

        return {
            "model_loaded": True,
            "config_path": str(config_path),
            "features_dir": str(features_dir),
            "labels_entries": len(dataset),
            "num_ready": len(ready_ids),
            "num_missing": len(missing),
            "missing": missing,
        }

    def run(
        self,
        samples: list[dict[str, Any]],
        *,
        dataset_root: Path,
        labels_file: Path,
        model_name: str = "quad-stream",
        skip_missing: bool = False,
        debug: bool = False,
        batch_size: int | None = None,
        num_workers: int | None = None,
        cpu_threads: int | None = None,
    ) -> list[dict[str, Any]]:
        quad_stream_dir, config_path = self._repo_paths()
        if str(quad_stream_dir) not in sys.path:
            sys.path.insert(0, str(quad_stream_dir))

        from src.data.dataset import DeepfakeDataset

        features_dir = self.features_dir or (dataset_root / "features")
        configured_threads = _configure_cpu_threads(cpu_threads, debug=debug)
        resolved_workers = (
            _default_num_workers(configured_threads) if num_workers is None else max(0, num_workers)
        )

        _debug_log(debug, "run: loading model")
        model = self._load_model(debug=debug)
        config = self._config

        entries = _entries_from_samples(samples)
        _debug_log(debug, f"run: building dataset features_dir={features_dir} entries={len(entries)}")
        dataset = DeepfakeDataset(
            dataset_root=dataset_root,
            features_dir=features_dir,
            entries=entries,
            skip_missing=skip_missing,
            debug=debug,
        )

        sample_lookup = {s["sample_id"]: s for s in samples}
        resolved_batch_size = batch_size or config["training"]["batch_size"]
        _debug_log(
            debug,
            f"run: creating dataloader batch_size={resolved_batch_size} "
            f"num_workers={resolved_workers} dataset_len={len(dataset)}",
        )

        loader_kwargs: dict[str, Any] = {
            "batch_size": resolved_batch_size,
            "shuffle": False,
            "num_workers": resolved_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        if resolved_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

        dataloader = DataLoader(dataset, **loader_kwargs)

        results: list[dict[str, Any]] = []
        batch_index = 0
        data_iter = iter(dataloader)
        with torch.no_grad(), tqdm(
            total=len(dataset),
            desc=f"{model_name} inference",
            unit="sample",
        ) as pbar:
            while True:
                fetch_started = time.perf_counter()
                if debug:
                    _debug_log(debug, f"batch {batch_index}: fetching from dataloader (feature load)")
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                fetch_ms = (time.perf_counter() - fetch_started) * 1000.0

                sample_id = batch["sample_id"]
                if isinstance(sample_id, (list, tuple)):
                    sample_ids = list(sample_id)
                else:
                    sample_ids = [sample_id]

                transfer_started = time.perf_counter()
                segment_stft = batch["segment_stft"].to(self._device)
                segment_logmel = batch["segment_logmel"].to(self._device)
                full_stft = batch["full_stft"].to(self._device)
                full_logmel = batch["full_logmel"].to(self._device)
                transfer_ms = (time.perf_counter() - transfer_started) * 1000.0

                infer_started = time.perf_counter()
                outputs = model(segment_stft, segment_logmel, full_stft, full_logmel).view(-1)
                probas = outputs.detach().cpu().numpy()
                infer_ms = (time.perf_counter() - infer_started) * 1000.0
                runtime_ms = infer_ms / max(len(sample_ids), 1)

                if debug:
                    _debug_log(
                        debug,
                        f"batch {batch_index}: sample_ids={sample_ids} "
                        f"fetch_ms={fetch_ms:.1f} transfer_ms={transfer_ms:.1f} "
                        f"infer_ms={infer_ms:.1f}",
                    )

                for sid, proba in zip(sample_ids, probas):
                    if sid not in sample_lookup:
                        continue
                    sample = sample_lookup[sid]
                    scores = scores_from_quad_stream(float(proba))
                    prediction = scores["prediction"]
                    results.append(
                        {
                            "sample_id": sid,
                            "model_name": model_name,
                            "prediction": prediction,
                            "confidence": scores["confidence"],
                            "real_score": scores["real_score"],
                            "deepfake_score": scores["deepfake_score"],
                            "error_type": compute_error_type(sample["ground_truth"], prediction),
                            "runtime_ms": runtime_ms,
                            "checkpoint_path": str(self.checkpoint_path),
                        }
                    )
                    pbar.update(1)
                batch_index += 1

        if skip_missing:
            processed_ids = {r["sample_id"] for r in results}
            missing = [s["sample_id"] for s in samples if s["sample_id"] not in processed_ids]
            if missing:
                print(f"Skipped {len(missing)} samples with missing precomputed features.")
        elif len(results) != len(dataset):
            processed_ids = {r["sample_id"] for r in results}
            missing = [s["sample_id"] for s in samples if s["sample_id"] not in processed_ids]
            raise FileNotFoundError(
                f"Missing precomputed features for {len(missing)} samples. "
                f"First missing sample_id: {missing[0] if missing else 'unknown'}"
            )

        results.sort(key=lambda row: row["sample_id"])
        return results
