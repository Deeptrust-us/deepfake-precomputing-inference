#!/usr/bin/env python3
"""Dataset analysis and verification for preprocessed audio (STFT/log-mel) model."""

import json
import os
from pathlib import Path
from collections import Counter

def analyze_audio_dataset():
    """
    Analyze dataset splits for an audio deepfake detection task.
    Assumes precomputed features (STFT, log-mel) and a labels.json per split.
    - Each entry in the JSON has: filename, label ("real" or "fake"), language, model_or_speaker.
    - Features are assumed to be 254x254 arrays.
    """
    data_root = "data/splits_8"
    print("=" * 80)
    print("AUDIO DATASET ANALYSIS")
    print("=" * 80)

    # Load splits by format 'labels_{split}.json', where split is in ['train', 'val', 'test']
    splits = {}
    for split in ['train', 'val', 'test']:
        split_path = Path(data_root) / f"labels_b_{split}.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Could not find {split_path}")
        with open(split_path, "r") as f:
            splits[split] = json.load(f)
    # Combine all splits for global stats if needed
    all_entries = []
    for split in ['train', 'val', 'test']:
        all_entries.extend(splits[split])
    splits["all"] = all_entries

    # Overall stats
    print("\n📊 OVERALL STATISTICS")
    print("-" * 80)
    total_samples = sum(len(splits[s]) for s in splits)
    print(f"Total Audio Samples: {total_samples:,}")
    for split in ['train', 'val', 'test']:
        count = len(splits[split])
        pct = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"  {split.capitalize():10s}: {count:5,} samples ({pct:5.1f}%)")

    # Class distribution ("real" or "fake"), plus optional: language, speaker/model
    print("\n📈 CLASS DISTRIBUTION")
    print("-" * 80)
    for split in ['train', 'val', 'test']:
        if not splits[split]:
            continue
        labels = [entry["label"].strip().lower() for entry in splits[split]]
        count_real = sum(1 for l in labels if l == "real")
        count_fake = sum(1 for l in labels if l == "fake")
        total = len(labels)
        print(f"\n{split.capitalize()} Split:")
        print(f"  Real samples: {count_real:5,} ({count_real/total*100:5.1f}%)")
        print(f"  Fake samples: {count_fake:5,} ({count_fake/total*100:5.1f}%)")
        imbalance = f"{count_fake/count_real:.2f}:1" if count_real > 0 else "N/A"
        print(f"  Imbalance ratio: {imbalance}")

    # Optional: Languages and Models/Speakers
    print("\n🔤 LANGUAGE & MODEL/SPEAKER DISTRIBUTION (Train split)")
    print("-" * 80)
    train_split = splits['train']
    if train_split:
        lang_counts = Counter(e.get("language", "unknown") for e in train_split)
        m_counts   = Counter(e.get("model_or_speaker", "unknown") for e in train_split if e.get("label") == "fake")
        sp_counts   = Counter(e.get("model_or_speaker", "unknown") for e in train_split if e.get("label") == "real")
        print("Languages:")
        for lang, cnt in lang_counts.most_common():
            print(f"  {lang:10s}: {cnt:5,} ({cnt/len(train_split)*100:4.1f}%)")
        print("Model:")
        for m, cnt in m_counts.most_common():
            print(f"  {m:16s}: {cnt:5,} ({cnt/len(train_split)*100:4.1f}%)")
        print("Speaker:")
        for sp, cnt in sp_counts.most_common():
            print(f"  {sp:16s}: {cnt:5,} ({cnt/len(train_split)*100:4.1f}%)")
    
    # Training sample calculation
    print("\n🎯 TRAINING DETAILS")
    print("-" * 80)
    train_count = len(train_split)
    batch_size = 32  # Or set from config if available
    batches_per_epoch = train_count // batch_size if batch_size > 0 else 0
    print(f"Training samples:  {train_count:,}")
    print(f"Batch size:        {batch_size}")
    print(f"Batches/epoch:     {batches_per_epoch:,}")
    print(f"Samples/epoch:     {train_count:,} (one random segment per audio per epoch)")
    print("\nNote: During training, a RANDOM segment (speech region) is used from each audio file each epoch.")
    print("      This boosts augmentation and diversity per epoch.")

    # Data pipeline summary, adapted for audio
    print("\n🔄 DATA PIPELINE SUMMARY")
    print("-" * 80)
    print("1. Audio .wav files exist under data/audio/ (flat or real/fake subfolders)")
    print("2. For each audio file:")
    print("     - Precompute and save features:")
    print("         - STFT image (1 x 254 x 254 numpy array)")
    print("         - log-mel spectrogram (1 x 254 x 254 numpy array)")
    print("     - Features saved under data/features/segment_stft/, segment_logmel/, full_stft/, full_logmel/")
    print("3. During training:")
    print("     - Resulting tensors [4x254x254]: segment_stft, segment_logmel, full_stft, full_logmel")
    print("     - Label: 0=real, 1=fake")
    print("4. During validation/test:")
    print("     - First speech segment, or the longest, is selected (deterministic)")

    print("\nEach sample provides:")
    print("   - segment_stft: [1, 254, 254] (float32)")
    print("   - segment_logmel: [1, 254, 254]")
    print("   - full_stft: [1, 254, 254]")
    print("   - full_logmel: [1, 254, 254]")
    print("   - label: 0 (real) or 1 (fake)")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    analyze_audio_dataset()

