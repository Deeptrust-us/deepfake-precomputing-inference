"""HM-Conformer batch inference runner."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from ..scoring import compute_error_type, scores_from_hm_conformer


class HmConformerRunner:
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self._handler = None

    def _load_handler(self):
        if self._handler is not None:
            return self._handler

        model_dir = self.checkpoint_path
        if model_dir.name == "params":
            model_dir = model_dir.parent

        repo_root = Path(__file__).resolve().parents[3]
        hm_conformer_dir = repo_root / "models" / "hm-conformer"
        if str(hm_conformer_dir) not in sys.path:
            sys.path.insert(0, str(hm_conformer_dir))

        from handler import EndpointHandler

        self._handler = EndpointHandler(path=str(model_dir))
        return self._handler

    def predict_file(self, audio_path: Path) -> dict[str, float]:
        handler = self._load_handler()
        result = handler({"inputs": str(audio_path)})
        if not result or "error" in result[0]:
            raise RuntimeError(result[0].get("error", "Unknown HM-Conformer inference error"))
        return {"deepfake_raw": float(result[0]["deepfake_score"])}

    def run(
        self,
        samples: list[dict[str, Any]],
        *,
        model_name: str = "hm-conformer",
        skip_missing: bool = False,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for sample in samples:
            audio_path = Path(sample["resolved_audio_path"])
            if not audio_path.exists():
                if skip_missing:
                    continue
                raise FileNotFoundError(f"Audio file not found for {sample['sample_id']}: {audio_path}")

            started = time.perf_counter()
            raw = self.predict_file(audio_path)
            runtime_ms = (time.perf_counter() - started) * 1000.0

            scores = scores_from_hm_conformer(raw["deepfake_raw"])
            ground_truth = sample["ground_truth"]
            prediction = scores["prediction"]
            results.append(
                {
                    "sample_id": sample["sample_id"],
                    "model_name": model_name,
                    "prediction": prediction,
                    "confidence": scores["confidence"],
                    "real_score": scores["real_score"],
                    "deepfake_score": scores["deepfake_score"],
                    "error_type": compute_error_type(ground_truth, prediction),
                    "runtime_ms": runtime_ms,
                    "checkpoint_path": str(self.checkpoint_path),
                }
            )
        return results
