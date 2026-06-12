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

    def _load_model(self):
        if self._model is not None:
            return self._model

        quad_stream_dir, config_path = self._repo_paths()
        if str(quad_stream_dir) not in sys.path:
            sys.path.insert(0, str(quad_stream_dir))

        from scripts.evaluate import load_checkpoint, _remap_state_dict_for_compat
        from src.models.quad_stream import QuadStreamModel

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        model_cfg = config.get("model", {})
        model = QuadStreamModel(
            backbone=model_cfg.get("backbone", "resnet18"),
            feature_dim=model_cfg.get("feature_dim", 256),
            fusion_dim=model_cfg.get("fusion_dim", 512),
            dropout=model_cfg.get("dropout", 0.5),
            pretrained=model_cfg.get("pretrained", True),
            use_attention=model_cfg.get("use_attention", False),
        ).to(self._device)

        checkpoint = load_checkpoint(
            str(self.checkpoint_path),
            self._device,
            trust_checkpoint=self.trust_checkpoint,
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            remapped, _ = _remap_state_dict_for_compat(state_dict)
            model.load_state_dict(remapped, strict=False)

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

        self._load_model()
        dataset = DeepfakeDataset(
            dataset_root=dataset_root,
            labels_file=labels_file,
            features_dir=features_dir,
        )

        ready_ids: list[str] = []
        missing: list[dict[str, str]] = []
        feature_kinds = ("segment_stft", "segment_logmel", "full_stft", "full_logmel")
        for sample in samples:
            stem = Path(sample["filename"]).stem
            try:
                for kind in feature_kinds:
                    dataset._feature_path(kind, stem)
                ready_ids.append(sample["sample_id"])
            except FileNotFoundError as exc:
                missing.append({"sample_id": sample["sample_id"], "error": str(exc)})

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
    ) -> list[dict[str, Any]]:
        quad_stream_dir, config_path = self._repo_paths()
        if str(quad_stream_dir) not in sys.path:
            sys.path.insert(0, str(quad_stream_dir))

        from src.data.dataset import DeepfakeDataset

        features_dir = self.features_dir or (dataset_root / "features")
        model = self._load_model()
        config = self._config

        dataset = DeepfakeDataset(
            dataset_root=dataset_root,
            labels_file=labels_file,
            features_dir=features_dir,
        )

        sample_lookup = {s["sample_id"]: s for s in samples}
        dataloader = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        results: list[dict[str, Any]] = []
        with torch.no_grad(), tqdm(
            total=len(samples),
            desc=f"{model_name} inference",
            unit="sample",
        ) as pbar:
            for batch in dataloader:
                sample_id = batch["sample_id"]
                if isinstance(sample_id, (list, tuple)):
                    sample_ids = list(sample_id)
                else:
                    sample_ids = [sample_id]

                filtered = [sid for sid in sample_ids if sid in sample_lookup]
                if not filtered:
                    continue

                started = time.perf_counter()
                segment_stft = batch["segment_stft"].to(self._device)
                segment_logmel = batch["segment_logmel"].to(self._device)
                full_stft = batch["full_stft"].to(self._device)
                full_logmel = batch["full_logmel"].to(self._device)

                outputs = model(segment_stft, segment_logmel, full_stft, full_logmel).view(-1)
                probas = outputs.detach().cpu().numpy()
                runtime_ms = (time.perf_counter() - started) * 1000.0 / max(len(sample_ids), 1)

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

        if skip_missing:
            processed_ids = {r["sample_id"] for r in results}
            missing = [s["sample_id"] for s in samples if s["sample_id"] not in processed_ids]
            if missing:
                print(f"Skipped {len(missing)} samples with missing precomputed features.")
        elif len(results) != len(samples):
            processed_ids = {r["sample_id"] for r in results}
            missing = [s["sample_id"] for s in samples if s["sample_id"] not in processed_ids]
            raise FileNotFoundError(
                f"Missing precomputed features for {len(missing)} samples. "
                f"First missing sample_id: {missing[0] if missing else 'unknown'}"
            )

        results.sort(key=lambda row: row["sample_id"])
        return results
