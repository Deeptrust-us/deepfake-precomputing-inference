"""Checkpoint loading helpers for QuadStream inference and evaluation."""

from __future__ import annotations

import pickle

import numpy as np
import torch


def _torch_load_compat(path: str, device, *, weights_only: bool):
    """torch.load wrapper compatible with older torch versions without weights_only."""
    try:
        return torch.load(path, map_location=device, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint(path: str, device, *, trust_checkpoint: bool = False):
    """Load a checkpoint with PyTorch 2.6+ safe defaults."""
    if trust_checkpoint:
        print("Loading checkpoint with weights_only=False (trusted checkpoint)")
        return _torch_load_compat(path, device, weights_only=False)

    try:
        return _torch_load_compat(path, device, weights_only=True)
    except pickle.UnpicklingError as e:
        try:
            if hasattr(torch, "serialization") and hasattr(torch.serialization, "safe_globals"):
                with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype]):
                    return _torch_load_compat(path, device, weights_only=True)
            if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype])
                return _torch_load_compat(path, device, weights_only=True)
        except Exception:
            pass

        raise RuntimeError(
            "Failed to load checkpoint safely (weights_only=True).\n"
            "If you trust this checkpoint (e.g., you trained it yourself), re-run with --trust-checkpoint "
            "to load with weights_only=False."
        ) from e


def remap_state_dict_for_compat(state_dict: dict) -> tuple[dict, list[str]]:
    """Best-effort remaps for older checkpoints when module layouts changed."""
    if not isinstance(state_dict, dict):
        return state_dict, []

    sd = dict(state_dict)
    notes: list[str] = []

    if ("fusion.0.weight" in sd or "fusion.0.bias" in sd) and (
        "fusion.1.weight" not in sd and "fusion.1.bias" not in sd
    ):
        for suffix in ("weight", "bias"):
            old_k = f"fusion.0.{suffix}"
            new_k = f"fusion.1.{suffix}"
            if old_k in sd and new_k not in sd:
                sd[new_k] = sd.pop(old_k)
        notes.append(
            "Remapped fusion.0.{weight,bias} -> fusion.1.{weight,bias} (added leading Dropout in fusion)."
        )

    if ("fusion.3.weight" in sd or "fusion.3.bias" in sd) and (
        "fusion.4.weight" not in sd and "fusion.4.bias" not in sd
    ):
        for suffix in ("weight", "bias"):
            old_k = f"fusion.3.{suffix}"
            new_k = f"fusion.4.{suffix}"
            if old_k in sd and new_k not in sd:
                sd[new_k] = sd.pop(old_k)
        notes.append(
            "Remapped fusion.3.{weight,bias} -> fusion.4.{weight,bias} (added leading Dropout in fusion)."
        )

    return sd, notes
