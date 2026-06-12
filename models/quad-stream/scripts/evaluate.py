"""Evaluation script for the audio quad-stream deepfake detection model.

Evaluates segment-level predictions (one per VAD segment) and also aggregates
predictions per audio file (`sample_id`) to compute file-level metrics.
"""

import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Add parent directory to path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.quad_stream import QuadStreamModel
from src.data.dataset import DeepfakeDataset
from src.utils.checkpoint import load_checkpoint, remap_state_dict_for_compat
from src.utils.metrics import compute_segment_metrics, compute_file_metrics


def evaluate_model(model, dataloader, device, aggregate_by_file=True):
    """Evaluate model on dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probas = []
    file_predictions = defaultdict(list)
    file_labels = {}
    
    with torch.no_grad():
        for batch in dataloader:
            segment_stft = batch['segment_stft'].to(device)
            segment_logmel = batch['segment_logmel'].to(device)
            full_stft = batch['full_stft'].to(device)
            full_logmel = batch['full_logmel'].to(device)
            labels = batch['label'].float().to(device)
            sample_ids = batch.get('sample_id')
            
            outputs = model(segment_stft, segment_logmel, full_stft, full_logmel)
            outputs = outputs.view(-1)  # safe for batch_size==1
            probas = outputs.detach().cpu().numpy()
            preds = (probas > 0.5).astype(int)
            labels_np = labels.detach().cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            all_probas.extend(probas)
            
            # Aggregate by audio file
            if aggregate_by_file and sample_ids is not None:
                for sid, proba, label in zip(sample_ids, probas, labels_np):
                    file_predictions[sid].append(float(proba))
                    file_labels[sid] = int(label)
    
    # Segment-level metrics
    frame_metrics = compute_segment_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probas)
    )
    
    # File-level metrics
    file_metrics = None
    if aggregate_by_file and len(file_predictions) > 0:
        file_metrics = compute_file_metrics(
            file_predictions,
            file_labels,
            aggregation="mean"
        )
    
    return frame_metrics, file_metrics, all_labels, all_preds, all_probas


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate audio quad-stream deepfake detection model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Load checkpoint with weights_only=False (unsafe if the checkpoint is untrusted).",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], 
                       help="Dataset split to evaluate")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results (default from config)")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = args.output_dir or config.get("paths", {}).get("results_dir", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    data_root = config['data']['data_root']
    split_key = f"{args.split}_labels_file"
    if split_key not in config.get("data", {}):
        raise KeyError(
            f"Missing config key data.{split_key} in {args.config}. "
            f"Set it to the labels JSON for the '{args.split}' split."
        )
    labels_file = config["data"][split_key]
    features_dir = config["data"]["features_dir"]

    if not os.path.exists(labels_file):
        raise FileNotFoundError(
            f"{split_key} not found: {labels_file}\n"
            f"Set data.{split_key} in {args.config} to an existing JSON file."
        )
    if not os.path.exists(features_dir):
        raise FileNotFoundError(
            f"features_dir not found: {features_dir}\n"
            f"Set data.features_dir in {args.config} to an existing directory."
        )
    print(f"Labels file: {labels_file}")
    print(f"Features dir: {features_dir}")

    dataset = DeepfakeDataset(
        dataset_root=data_root,
        labels_file=labels_file,
        features_dir=features_dir,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    # Initialize model
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    model = QuadStreamModel(
        backbone=model_cfg.get("backbone", "resnet18"),
        feature_dim=model_cfg.get("feature_dim", 256),
        fusion_dim=model_cfg.get("fusion_dim", 512),
        dropout=model_cfg.get("dropout", 0.5),
        pretrained=model_cfg.get("pretrained", True),
        use_attention=model_cfg.get("use_attention", False),
    ).to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, device, trust_checkpoint=bool(args.trust_checkpoint))
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Try strict load first; if it fails due to known layout changes, remap and retry.
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        remapped, notes = remap_state_dict_for_compat(state_dict)
        if notes:
            print("Checkpoint compatibility remap applied:")
            for n in notes:
                print(f"  - {n}")
        try:
            model.load_state_dict(remapped, strict=True)
        except RuntimeError:
            # Last resort: load non-strict but report what was ignored.
            incompatible = model.load_state_dict(remapped, strict=False)
            print("WARNING: Loaded checkpoint with strict=False due to remaining key mismatches.")
            if getattr(incompatible, "missing_keys", None):
                print(f"  Missing keys ({len(incompatible.missing_keys)}): {incompatible.missing_keys[:20]}")
                if len(incompatible.missing_keys) > 20:
                    print("  ...")
            if getattr(incompatible, "unexpected_keys", None):
                print(f"  Unexpected keys ({len(incompatible.unexpected_keys)}): {incompatible.unexpected_keys[:20]}")
                if len(incompatible.unexpected_keys) > 20:
                    print("  ...")
            # Re-raise original error if nothing loaded (very unlikely), but keep going otherwise.
            if len(getattr(incompatible, "missing_keys", [])) == 0 and len(getattr(incompatible, "unexpected_keys", [])) == 0:
                raise e
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Evaluate
    print(f"Evaluating on {args.split} split...")
    frame_metrics, file_metrics, y_true, y_pred, y_proba = evaluate_model(
        model, dataloader, device, aggregate_by_file=True
    )
    
    # Print results
    print("\n" + "="*50)
    print(f"Segment-level Metrics ({args.split}):")
    print("="*50)
    for key, value in frame_metrics.items():
        print(f"  {key.upper()}: {value:.4f}")
    
    if file_metrics:
        print("\n" + "="*50)
        print(f"File-level Metrics ({args.split}):")
        print("="*50)
        for key, value in file_metrics.items():
            print(f"  {key.upper()}: {value:.4f}")
    
    # Save results
    results_file = os.path.join(output_dir, f"{args.split}_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Segment-level Metrics ({args.split}):\n")
        for key, value in frame_metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        
        if file_metrics:
            f.write(f"\nFile-level Metrics ({args.split}):\n")
            for key, value in file_metrics.items():
                f.write(f"{key}: {value:.4f}\n")
    
    # Plot confusion matrix
    cm_path = os.path.join(output_dir, f"{args.split}_confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, cm_path)
    print(f"\nConfusion matrix saved to {cm_path}")
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()


