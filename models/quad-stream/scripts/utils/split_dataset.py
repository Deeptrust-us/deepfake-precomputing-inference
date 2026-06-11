"""
Split one labels.json into 8 JSON label files:

Step A) Split into two "halves" by UNIQUE model_or_speaker values, separately for:
  - fake entries (model_or_speaker = model)
  - real entries (model_or_speaker = speaker/source)

This produces:
  - labels_a.json
  - labels_b.json

Step B) For each of (A,B), split into train/val/test stratified by (label, model_or_speaker):
  - labels_a_train.json / labels_a_val.json / labels_a_test.json
  - labels_b_train.json / labels_b_val.json / labels_b_test.json

Additionally writes:
  - splits_metadata.md  (human-readable manifest; NOT a JSON file to keep training-compatible)

Usage:
  source venv/bin/active
  python scripts/utils/split_labels_8.py --labels-path data/labels.json --out-dir data/splits_8 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _norm_label(x: Any) -> str:
    return str(x or "").strip().lower()


def _norm_mos(x: Any) -> str:
    s = str(x or "").strip()
    return s if s else "unknown"


def _load_entries(labels_path: Path) -> List[Dict[str, Any]]:
    """
    Load labels.json entries.
    - If ijson is available, stream parse to reduce peak memory.
    - Otherwise, fall back to json.load (loads entire list into memory).
    """
    try:
        import ijson  # type: ignore

        entries: List[Dict[str, Any]] = []
        with labels_path.open("rb") as f:
            for item in ijson.items(f, "item"):
                if isinstance(item, dict):
                    entries.append(item)
        return entries
    except Exception:
        raw = json.loads(labels_path.read_text())
        if not isinstance(raw, list):
            raise ValueError(f"{labels_path} must be a JSON list")
        return raw  # type: ignore[return-value]


def _write_json_list(path: Path, entries: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))


def _split_counts(n: int, train: float, val: float, test: float) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    total = train + val + test
    if total <= 0:
        raise ValueError("train/val/test ratios must sum to > 0")
    train, val, test = train / total, val / total, test / total

    # Largest-remainder method to keep sums exact.
    raw = [train * n, val * n, test * n]
    floors = [int(math.floor(x)) for x in raw]
    remainder = n - sum(floors)
    fracs = [(raw[i] - floors[i], i) for i in range(3)]
    fracs.sort(reverse=True)
    for k in range(remainder):
        floors[fracs[k][1]] += 1
    return floors[0], floors[1], floors[2]


def _stable_shuffle(items: List[Dict[str, Any]], rng: random.Random) -> None:
    # Deterministic shuffle but tries to avoid depending on dict ordering:
    # pre-sort on a stable key, then shuffle.
    def key_fn(e: Dict[str, Any]) -> Tuple[str, str, str]:
        return (
            str(e.get("id", "")),
            str(e.get("filename", "")),
            str(e.get("original_path", "")),
        )

    items.sort(key=key_fn)
    rng.shuffle(items)


def _split_halves(values: List[str], rng: random.Random, strategy: str) -> Tuple[List[str], List[str]]:
    vals = list(values)
    if strategy == "random":
        rng.shuffle(vals)
    else:
        vals.sort()
    mid = (len(vals) + 1) // 2  # left gets the extra if odd
    return vals[:mid], vals[mid:]


def _stratified_split(
    entries: List[Dict[str, Any]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    strata: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for e in entries:
        label = _norm_label(e.get("label"))
        mos = _norm_mos(e.get("model_or_speaker"))
        strata[(label, mos)].append(e)

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    # Sort keys for deterministic iteration, then shuffle per stratum with derived seeds.
    for (label, mos) in sorted(strata.keys()):
        bucket = strata[(label, mos)]
        derived_seed = (seed * 1_000_003) ^ (hash(label) & 0xFFFFFFFF) ^ ((hash(mos) & 0xFFFFFFFF) << 1)
        brng = random.Random(derived_seed)
        _stable_shuffle(bucket, brng)

        n_tr, n_val, n_te = _split_counts(len(bucket), train_ratio, val_ratio, test_ratio)
        train.extend(bucket[:n_tr])
        val.extend(bucket[n_tr : n_tr + n_val])
        test.extend(bucket[n_tr + n_val : n_tr + n_val + n_te])

    _stable_shuffle(train, rng)
    _stable_shuffle(val, rng)
    _stable_shuffle(test, rng)
    return train, val, test


def _unique_mos(entries: Iterable[Dict[str, Any]]) -> List[str]:
    return sorted({_norm_mos(e.get("model_or_speaker")) for e in entries})


def _label_counts(entries: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {"real": 0, "fake": 0, "other": 0}
    for e in entries:
        lab = _norm_label(e.get("label"))
        if lab == "real":
            out["real"] += 1
        elif lab == "fake":
            out["fake"] += 1
        else:
            out["other"] += 1
    return out


def _format_list(xs: List[str], max_items: int = 200) -> str:
    if len(xs) <= max_items:
        return ", ".join(xs) if xs else "(none)"
    head = ", ".join(xs[:max_items])
    return f"{head}, ... (+{len(xs) - max_items} more)"


def main() -> None:
    p = argparse.ArgumentParser(description="Split labels.json into 8 JSON label files + a metadata manifest.")
    p.add_argument("--labels-path", type=str, required=True, help="Path to input labels.json")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for 8 JSON files")
    p.add_argument("--seed", type=int, default=42, help="Random seed (used for random half-split and shuffling)")
    p.add_argument("--half-strategy", choices=["sorted", "random"], default="sorted", help="How to split unique models/speakers into halves")
    p.add_argument("--train", type=float, default=0.8, help="Train ratio")
    p.add_argument("--val", type=float, default=0.1, help="Val ratio")
    p.add_argument("--test", type=float, default=0.1, help="Test ratio")
    args = p.parse_args()

    labels_path = Path(args.labels_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries_all = _load_entries(labels_path)
    if not entries_all:
        raise ValueError("labels.json is empty")

    fake_entries = [e for e in entries_all if _norm_label(e.get("label")) == "fake"]
    real_entries = [e for e in entries_all if _norm_label(e.get("label")) == "real"]
    other_entries = [e for e in entries_all if _norm_label(e.get("label")) not in ("fake", "real")]

    rng_models = random.Random(args.seed + 101)
    rng_speakers = random.Random(args.seed + 202)
    fake_models = _unique_mos(fake_entries)
    real_speakers = _unique_mos(real_entries)
    fake_a, fake_b = _split_halves(fake_models, rng_models, args.half_strategy)
    real_a, real_b = _split_halves(real_speakers, rng_speakers, args.half_strategy)
    set_fake_a, set_fake_b = set(fake_a), set(fake_b)
    set_real_a, set_real_b = set(real_a), set(real_b)

    # Group A / B: each gets half of fake models and half of real speakers.
    group_a: List[Dict[str, Any]] = []
    group_b: List[Dict[str, Any]] = []
    for e in entries_all:
        lab = _norm_label(e.get("label"))
        mos = _norm_mos(e.get("model_or_speaker"))
        if lab == "fake":
            (group_a if mos in set_fake_a else group_b).append(e)
        elif lab == "real":
            (group_a if mos in set_real_a else group_b).append(e)
        else:
            # Keep unknown labels, but don't duplicate; stick them in A.
            group_a.append(e)

    # Write the two "all" halves (these are part of the requested 8 JSON files).
    labels_a_path = out_dir / "labels_a.json"
    labels_b_path = out_dir / "labels_b.json"
    _write_json_list(labels_a_path, group_a)
    _write_json_list(labels_b_path, group_b)

    # Split each half into train/val/test.
    a_train, a_val, a_test = _stratified_split(group_a, args.seed + 1, args.train, args.val, args.test)
    b_train, b_val, b_test = _stratified_split(group_b, args.seed + 2, args.train, args.val, args.test)

    _write_json_list(out_dir / "labels_a_train.json", a_train)
    _write_json_list(out_dir / "labels_a_val.json", a_val)
    _write_json_list(out_dir / "labels_a_test.json", a_test)
    _write_json_list(out_dir / "labels_b_train.json", b_train)
    _write_json_list(out_dir / "labels_b_val.json", b_val)
    _write_json_list(out_dir / "labels_b_test.json", b_test)

    # Human-readable metadata manifest (NOT json; avoids breaking code expecting JSON lists).
    md_path = out_dir / "splits_metadata.md"
    md_lines: List[str] = []
    md_lines.append("## labels.json 8-way split metadata\n")
    md_lines.append(f"- Input: `{labels_path}`\n")
    md_lines.append(f"- Output dir: `{out_dir}`\n")
    md_lines.append(f"- Seed: `{args.seed}`\n")
    md_lines.append(f"- Half strategy: `{args.half_strategy}`\n")
    md_lines.append(f"- Ratios: train={args.train}, val={args.val}, test={args.test}\n")
    if other_entries:
        md_lines.append(f"- NOTE: Found {len(other_entries)} entries with non-(real|fake) labels; assigned to group A.\n")
    md_lines.append("\n### Group composition (unique model_or_speaker)\n")
    md_lines.append(f"- Fake models A ({len(fake_a)}): {_format_list(fake_a)}\n")
    md_lines.append(f"- Fake models B ({len(fake_b)}): {_format_list(fake_b)}\n")
    md_lines.append(f"- Real speakers A ({len(real_a)}): {_format_list(real_a)}\n")
    md_lines.append(f"- Real speakers B ({len(real_b)}): {_format_list(real_b)}\n")

    def add_file_section(name: str, entries: List[Dict[str, Any]]) -> None:
        counts = _label_counts(entries)
        f_models = sorted({_norm_mos(e.get("model_or_speaker")) for e in entries if _norm_label(e.get("label")) == "fake"})
        r_speakers = sorted({_norm_mos(e.get("model_or_speaker")) for e in entries if _norm_label(e.get("label")) == "real"})
        md_lines.append(f"\n### `{name}`\n")
        md_lines.append(f"- Entries: {len(entries)} (real={counts['real']}, fake={counts['fake']}, other={counts['other']})\n")
        md_lines.append(f"- Fake models present ({len(f_models)}): {_format_list(f_models)}\n")
        md_lines.append(f"- Real speakers present ({len(r_speakers)}): {_format_list(r_speakers)}\n")

    add_file_section("labels_a.json", group_a)
    add_file_section("labels_b.json", group_b)
    add_file_section("labels_a_train.json", a_train)
    add_file_section("labels_a_val.json", a_val)
    add_file_section("labels_a_test.json", a_test)
    add_file_section("labels_b_train.json", b_train)
    add_file_section("labels_b_val.json", b_val)
    add_file_section("labels_b_test.json", b_test)

    md_path.write_text("".join(md_lines), encoding="utf-8")

    print(f"✓ Wrote 8 JSON files to: {out_dir}")
    print(f"✓ Wrote metadata manifest: {md_path}")


if __name__ == "__main__":
    main()

