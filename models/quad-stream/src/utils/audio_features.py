"""Audio feature extraction utilities (STFT + log-mel) for 1x224x224 inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import torchaudio
except Exception as e:  # pragma: no cover
    torchaudio = None  # type: ignore


@dataclass(frozen=True)
class AudioFeatureConfig:
    sample_rate: int = 16000
    segment_seconds: float = 2.0
    max_full_seconds: Optional[float] = None  # if set, crop full audio to this duration

    # STFT parameters
    stft_n_fft: int = 1024
    stft_hop_length: int = 160
    stft_win_length: int = 400

    # Mel parameters
    mel_n_fft: int = 1024
    mel_hop_length: int = 160
    mel_win_length: int = 400
    mel_n_mels: int = 224
    mel_f_min: float = 0.0
    mel_f_max: Optional[float] = None

    # Output
    out_size: int = 224


_SILERO_VAD_MODEL: Optional[torch.nn.Module] = None
_SILERO_VAD_AVAILABLE: Optional[bool] = None

# Cache transforms because constructing them for every segment/file is slow.
_SPECTROGRAM_CACHE: dict[tuple, "torchaudio.transforms.Spectrogram"] = {}  # type: ignore[name-defined]
_MEL_CACHE: dict[tuple, "torchaudio.transforms.MelSpectrogram"] = {}  # type: ignore[name-defined]


def load_audio_mono(path: str | Path, target_sr: int) -> Tuple[torch.Tensor, int]:
    """Load WAV and return mono waveform (1, T) at target sample rate."""
    if torchaudio is None:
        raise RuntimeError("torchaudio is required to load audio files.")

    wav, sr = torchaudio.load(str(path))  # (C, T)
    if wav.ndim != 2:
        raise ValueError(f"Unexpected waveform shape: {tuple(wav.shape)}")

    # Convert to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    return wav, sr


def _resize_2d_to_square(x: torch.Tensor, out_size: int) -> torch.Tensor:
    """Resize (F, T) or (1, F, T) -> (1, out, out) using bilinear interpolation."""
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError(f"Expected 2D/3D tensor, got shape {tuple(x.shape)}")

    x = x.unsqueeze(0)  # (N=1, C=1, H, W)
    x = F.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False)
    return x.squeeze(0)  # (1, out, out)


def _safe_log(x: torch.Tensor) -> torch.Tensor:
    # log1p keeps stability for small values and avoids -inf for 0
    return torch.log1p(torch.clamp(x, min=0.0))


def _zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean()
    std = x.std().clamp(min=eps)
    return (x - mean) / std


def _get_silero_vad_model() -> Optional[torch.nn.Module]:
    """Load Silero VAD model once per process (if installed), else None.

    Note: We intentionally keep VAD on CPU in this codebase for robustness.
    GPU VAD is possible, but Silero helpers often assume CPU tensors unless
    carefully configured.
    """
    global _SILERO_VAD_MODEL, _SILERO_VAD_AVAILABLE  # noqa: PLW0603

    if _SILERO_VAD_AVAILABLE is False:
        return None
    if _SILERO_VAD_MODEL is not None:
        return _SILERO_VAD_MODEL

    try:
        from silero_vad import load_silero_vad  # type: ignore
    except Exception:
        _SILERO_VAD_AVAILABLE = False
        return None

    model = load_silero_vad()
    try:
        model.eval()
    except Exception:
        pass

    _SILERO_VAD_MODEL = model
    _SILERO_VAD_AVAILABLE = True
    return _SILERO_VAD_MODEL


def _try_silero_vad_segments(wav: torch.Tensor, sr: int) -> Optional[list[tuple[int, int]]]:
    """Return list of (start_sample, end_sample) speech segments if Silero VAD is available."""
    model = _get_silero_vad_model()
    if model is None:
        return None

    try:
        from silero_vad import get_speech_timestamps  # type: ignore
    except Exception:
        return None

    # Silero expects 1D float32 tensor at 8k or 16k (we use 16k)
    wav_1d = wav.squeeze(0).detach().cpu().to(torch.float32)
    if sr not in (8000, 16000):
        return None

    timestamps = get_speech_timestamps(wav_1d, model, sampling_rate=sr)
    if not timestamps:
        return []

    segments: list[tuple[int, int]] = []
    for seg in timestamps:
        start = int(seg["start"])
        end = int(seg["end"])
        if end > start:
            segments.append((start, end))
    return segments


def get_speech_segments(wav: torch.Tensor, sr: int) -> list[tuple[int, int]]:
    """Return speech segments as (start_sample, end_sample).

    - If Silero VAD is available, returns all detected speech segments.
    - If VAD is unavailable or no speech is detected, returns a single segment covering the whole audio.
    """
    segments = _try_silero_vad_segments(wav, sr)
    if segments is None:
        return [(0, wav.shape[-1])]
    if len(segments) == 0:
        return [(0, wav.shape[-1])]
    return segments


def _get_spectrogram_transform(cfg: AudioFeatureConfig, device: torch.device) -> "torchaudio.transforms.Spectrogram":  # type: ignore[name-defined]
    if torchaudio is None:
        raise RuntimeError("torchaudio is required to compute STFT features.")
    key = (
        "spec",
        cfg.stft_n_fft,
        cfg.stft_hop_length,
        cfg.stft_win_length,
        2.0,
        True,
        "reflect",
        str(device),
    )
    tr = _SPECTROGRAM_CACHE.get(key)
    if tr is None:
        tr = torchaudio.transforms.Spectrogram(  # type: ignore[attr-defined]
            n_fft=cfg.stft_n_fft,
            hop_length=cfg.stft_hop_length,
            win_length=cfg.stft_win_length,
            power=2.0,
            center=True,
            pad_mode="reflect",
        ).to(device)
        _SPECTROGRAM_CACHE[key] = tr
    return tr


def _get_mel_transform(cfg: AudioFeatureConfig, device: torch.device) -> "torchaudio.transforms.MelSpectrogram":  # type: ignore[name-defined]
    if torchaudio is None:
        raise RuntimeError("torchaudio is required to compute mel features.")
    key = (
        "mel",
        cfg.sample_rate,
        cfg.mel_n_fft,
        cfg.mel_hop_length,
        cfg.mel_win_length,
        cfg.mel_n_mels,
        float(cfg.mel_f_min),
        float(cfg.mel_f_max) if cfg.mel_f_max is not None else None,
        2.0,
        True,
        "reflect",
        str(device),
    )
    tr = _MEL_CACHE.get(key)
    if tr is None:
        tr = torchaudio.transforms.MelSpectrogram(  # type: ignore[attr-defined]
            sample_rate=cfg.sample_rate,
            n_fft=cfg.mel_n_fft,
            hop_length=cfg.mel_hop_length,
            win_length=cfg.mel_win_length,
            n_mels=cfg.mel_n_mels,
            f_min=cfg.mel_f_min,
            f_max=cfg.mel_f_max,
            power=2.0,
            center=True,
            pad_mode="reflect",
        ).to(device)
        _MEL_CACHE[key] = tr
    return tr


def extract_fixed_length_segment(
    wav: torch.Tensor,
    sr: int,
    start_sample: int,
    end_sample: int,
    segment_seconds: float,
) -> torch.Tensor:
    """Extract a segment from [start_sample:end_sample] and center crop/pad to fixed duration."""
    seg_len = max(int(segment_seconds * sr), 1)
    seg = wav[:, max(start_sample, 0) : max(end_sample, 0)]

    if seg.shape[-1] >= seg_len:
        s = max((seg.shape[-1] - seg_len) // 2, 0)
        seg = seg[:, s : s + seg_len]
    else:
        pad = seg_len - seg.shape[-1]
        left = pad // 2
        right = pad - left
        seg = F.pad(seg, (left, right))

    return seg


def compute_stft_image(wav: torch.Tensor, cfg: AudioFeatureConfig) -> torch.Tensor:
    """Compute log-magnitude STFT and resize to (1, 224, 224)."""
    device = wav.device
    spec_tr = _get_spectrogram_transform(cfg, device)
    spec = spec_tr(wav)  # (1, F, T)

    x = _safe_log(spec)
    x = _zscore(x)
    return _resize_2d_to_square(x, cfg.out_size)  # (1, 224, 224)


def compute_logmel_image(wav: torch.Tensor, cfg: AudioFeatureConfig) -> torch.Tensor:
    """Compute log-mel spectrogram and resize to (1, 224, 224)."""
    device = wav.device
    mel_tr = _get_mel_transform(cfg, device)
    mel = mel_tr(wav)  # (1, M=224, T)

    x = _safe_log(mel)
    x = _zscore(x)
    return _resize_2d_to_square(x, cfg.out_size)  # (1, 224, 224)


def compute_all_four_features(
    wav: torch.Tensor,
    cfg: AudioFeatureConfig,
) -> dict[str, torch.Tensor]:
    """Return the 4-stream audio inputs as float32 tensors (1, 224, 224).

    Note: this helper picks a single segment (index 0 / whole audio if no VAD),
    and is mainly intended for quick experiments. For training over *all* VAD
    segments, prefer `get_speech_segments(...)` + `extract_fixed_length_segment(...)`.
    """
    # Full audio optionally cropped
    full = wav
    if cfg.max_full_seconds is not None:
        max_len = int(cfg.max_full_seconds * cfg.sample_rate)
        if full.shape[-1] > max_len:
            full = full[:, :max_len]

    segments = get_speech_segments(full, cfg.sample_rate)
    s0, e0 = segments[0]
    segment = extract_fixed_length_segment(full, cfg.sample_rate, s0, e0, cfg.segment_seconds)

    return {
        "segment_stft": compute_stft_image(segment, cfg).to(torch.float32),
        "segment_logmel": compute_logmel_image(segment, cfg).to(torch.float32),
        "full_stft": compute_stft_image(full, cfg).to(torch.float32),
        "full_logmel": compute_logmel_image(full, cfg).to(torch.float32),
    }


def save_feature_npy(path: str | Path, x: torch.Tensor) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = x.detach().cpu().numpy().astype(np.float32)
    np.save(str(path), arr)

