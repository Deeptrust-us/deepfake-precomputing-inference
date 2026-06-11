"""Training script for the audio quad-stream deepfake detection model."""

import os
import sys
import argparse
import logging
import time
from datetime import datetime
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Add parent directory to path to import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.quad_stream import QuadStreamModel
from src.data.dataset import DeepfakeDataset
from src.utils.metrics import compute_segment_metrics

def _now() -> str:
    """Human-friendly timestamp for logs."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _metrics_to_python_floats(metrics: dict) -> dict:
    """Convert metric values (including numpy scalars) into plain Python floats for safe checkpointing."""
    out = {}
    for k, v in (metrics or {}).items():
        try:
            out[str(k)] = float(v)
        except Exception:
            # Fallback: keep original if it can't be converted
            out[str(k)] = v
    return out


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, class_weights=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probas = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        segment_stft = batch['segment_stft'].to(device)
        segment_logmel = batch['segment_logmel'].to(device)
        full_stft = batch['full_stft'].to(device)
        full_logmel = batch['full_logmel'].to(device)
        labels = batch['label'].float().to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(segment_stft, segment_logmel, full_stft, full_logmel).squeeze()
        
        # Compute loss
        if isinstance(criterion, nn.BCELoss) and criterion.reduction == 'none':
            # Weighted BCE loss
            per_sample_loss = criterion(outputs, labels)
            sample_weights = class_weights[labels.long()]
            loss = (per_sample_loss * sample_weights).mean()
        else:
            # Focal loss or standard loss (already handles reduction)
            loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        preds = (outputs > 0.5).cpu().numpy()
        probas = outputs.detach().cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels_np)
        all_probas.extend(probas)
        
        pbar.set_postfix({'loss': loss.item()})
    
    # Compute epoch metrics
    metrics = compute_segment_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probas)
    )
    metrics['loss'] = running_loss / len(dataloader)
    
    return metrics


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probas = []
    
    # Use standard BCE loss for validation (not weighted)
    val_criterion = nn.BCELoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            segment_stft = batch['segment_stft'].to(device)
            segment_logmel = batch['segment_logmel'].to(device)
            full_stft = batch['full_stft'].to(device)
            full_logmel = batch['full_logmel'].to(device)
            labels = batch['label'].float().to(device)
            
            outputs = model(segment_stft, segment_logmel, full_stft, full_logmel).squeeze()
            loss = val_criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = (outputs > 0.5).cpu().numpy()
            probas = outputs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            all_probas.extend(probas)
    
    metrics = compute_segment_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probas)
    )
    metrics['loss'] = running_loss / len(dataloader)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train audio quad-stream deepfake detection model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("train")
    run_start = time.time()
    log.info("Run started")
    
    # Load config
    log.info("Loading config: %s", args.config)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)
    
    # Create directories
    log.info("Creating output dirs (checkpoints/logs)")
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # Create datasets
    data_root = config['data']['data_root']
    train_labels_file = config["data"]["train_labels_file"]
    val_labels_file = config["data"]["val_labels_file"]
    features_dir = config["data"]["features_dir"]

    log.info("Data root:     %s", data_root)
    log.info("Train labels:  %s", train_labels_file)
    log.info("Val labels:    %s", val_labels_file)
    log.info("Features dir:  %s", features_dir)

    log.info("Building train dataset...")
    train_dataset = DeepfakeDataset(
        dataset_root=data_root,
        labels_file=train_labels_file,
        features_dir=features_dir,
    )
    log.info("Train dataset ready: %d samples", len(train_dataset))
    
    # Check if validation set exists and has data
    if os.path.exists(val_labels_file):
        log.info("Building validation dataset...")
        val_dataset = DeepfakeDataset(
            dataset_root=data_root,
            labels_file=val_labels_file,
            features_dir=features_dir,
        )
        log.info("Val dataset ready: %d samples", len(val_dataset))
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']
        ) if len(val_dataset) > 0 else None
    else:
        val_loader = None
        val_dataset = None
    
    # Create dataloaders with optional oversampling for real videos
    oversample_real = config['training'].get('oversample_real', False)
    if oversample_real:
        from torch.utils.data import WeightedRandomSampler
        # Calculate sample weights: higher weight for real audio (label=0)
        real_oversample_ratio = config['training'].get('real_oversample_ratio', 2.0)
        sample_weights = []
        for entry in train_dataset.entries:
            y = 0 if str(entry.get("label", "")).strip().lower() == "real" else 1
            sample_weights.append(real_oversample_ratio if y == 0 else 1.0)
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        log.info("Using WeightedRandomSampler (real oversample ratio: %.2fx)", real_oversample_ratio)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            sampler=sampler,  # Use sampler instead of shuffle
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']
        )
    else:
        log.info("Using standard shuffled DataLoader for training")
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']
        )
    
    # Initialize model (audio quad-stream)
    log.info("Initializing model...")
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    model = QuadStreamModel(
        backbone=model_cfg.get("backbone", "resnet18"),
        feature_dim=model_cfg.get("feature_dim", 256),
        fusion_dim=model_cfg.get("fusion_dim", 512),
        dropout=model_cfg.get("dropout", 0.5),
        pretrained=model_cfg.get("pretrained", True),
        use_attention=model_cfg.get("use_attention", False),
        freeze_stft_backbones=bool(model_cfg.get("freeze_stft_backbones", False)),
        freeze_logmel_backbones=bool(model_cfg.get("freeze_logmel_backbones", False)),
    ).to(device)
    
    # Freeze backbones if requested
    if bool(model_cfg.get("freeze_backbones", False)):
        model.freeze_all_backbones()
        log.info("Backbones FROZEN (all 4 streams; only per-stream FC + fusion + classifier trainable)")
    else:
        freeze_stft = bool(model_cfg.get("freeze_stft_backbones", False))
        freeze_logmel = bool(model_cfg.get("freeze_logmel_backbones", False))

        if freeze_stft:
            model.freeze_stft_backbones()
        if freeze_logmel:
            model.freeze_logmel_backbones()

        if freeze_stft and freeze_logmel:
            log.info("Backbones FROZEN (STFT + Log-Mel streams; all 4 stream backbones frozen)")
        elif freeze_stft:
            log.info("Backbones FROZEN (STFT streams only; segment_stft + full_stft backbones frozen)")
        elif freeze_logmel:
            log.info("Backbones FROZEN (Log-Mel streams only; segment_logmel + full_logmel backbones frozen)")
        else:
            log.info("Backbones UNFROZEN (all 4 streams; all parameters trainable)")
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    log.info("Params total=%s trainable=%s frozen=%s", f"{total_params:,}", f"{trainable_params:,}", f"{frozen_params:,}")
    
    # Loss and optimizer
    # Calculate class weights for imbalanced dataset
    log.info("Computing class weights...")
    train_labels = []
    for entry in train_dataset.entries:
        train_labels.append(0 if str(entry.get("label", "")).strip().lower() == "real" else 1)
    class_counts = np.bincount(train_labels)
    total = len(train_labels)
    
    # Calculate weights: inverse frequency (more weight to minority class)
    class_weights = total / (len(class_counts) * class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    log.info("Class distribution: Real=%d Fake=%d", int(class_counts[0]), int(class_counts[1]))
    log.info("Class weights: Real=%.4f Fake=%.4f", float(class_weights[0]), float(class_weights[1]))
    
    # Choose loss function
    use_focal_loss = config['training'].get('use_focal_loss', False)
    if use_focal_loss:
        from src.utils.focal_loss import FocalLoss
        focal_gamma = config['training'].get('focal_loss_gamma', 2.0)
        # Use inverse frequency as alpha for positive class
        alpha_fake = class_weights[1] / (class_weights[0] + class_weights[1])
        criterion = FocalLoss(alpha=alpha_fake, gamma=focal_gamma, reduction='mean')
        log.info("Using Focal Loss (alpha=%.4f gamma=%.2f)", float(alpha_fake), float(focal_gamma))
    else:
        log.info("Using weighted BCE loss to handle class imbalance")
        # Use standard BCE loss with no reduction - we'll apply weights manually in training loop
        criterion = nn.BCELoss(reduction='none')  # No reduction, we'll weight manually
    
    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if config['training']['optimizer'] == "adam":
        log.info("Optimizer: Adam (lr=%s wd=%s)", str(config['training']['learning_rate']), str(config['training']['weight_decay']))
        optimizer = optim.Adam(
            trainable_params,
            lr=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay'])
        )
    else:
        log.info(
            "Optimizer: SGD (lr=%s mom=%s wd=%s)",
            str(config['training']['learning_rate']),
            str(config['training']['momentum']),
            str(config['training']['weight_decay']),
        )
        optimizer = optim.SGD(
            trainable_params,
            lr=float(config['training']['learning_rate']),
            momentum=float(config['training']['momentum']),
            weight_decay=float(config['training']['weight_decay'])
        )
    
    # Learning rate scheduler
    if config['training']['scheduler'] == "cosine":
        log.info("Scheduler: CosineAnnealingLR (T_max=%s)", str(config['training']['num_epochs']))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
    else:
        log.info(
            "Scheduler: StepLR (step_size=%s gamma=%s)",
            str(config['training']['scheduler_params']['step_size']),
            str(config['training']['scheduler_params']['gamma']),
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training']['scheduler_params']['step_size'],
            gamma=config['training']['scheduler_params']['gamma']
        )
    
    # TensorBoard writer
    log.info("TensorBoard log dir: %s", config['paths']['log_dir'])
    writer = SummaryWriter(log_dir=config['paths']['log_dir'])
    
    # Training loop
    best_val_auc = 0.0
    patience_counter = 0
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        log.info("Resuming from checkpoint: %s", args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_auc = float(checkpoint.get('best_val_auc', 0.0))
        log.info("Resumed from epoch %d", int(start_epoch))
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        epoch_start = time.time()
        log.info("Epoch %d/%d started", epoch + 1, int(config['training']['num_epochs']))
        log.info("Current LR: %.6g", float(optimizer.param_groups[0]['lr']))

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, class_weights=class_weights_tensor)
        
        # Validate
        if val_loader is not None and len(val_dataset) > 0:
            val_metrics = validate(model, val_loader, criterion, device)
        else:
            # Use train metrics as validation if no validation set
            val_metrics = train_metrics.copy()
            log.warning("No validation set available; using training metrics as validation")
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        for key, value in train_metrics.items():
            writer.add_scalar(f'Train/{key}', value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f'Val/{key}', value, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        log.info(
            "Epoch %d summary | train: loss=%.4f auc=%.4f f1=%.4f | val: loss=%.4f auc=%.4f f1=%.4f",
            epoch + 1,
            float(train_metrics.get("loss", 0.0)),
            float(train_metrics.get("auc", 0.0)),
            float(train_metrics.get("f1", 0.0)),
            float(val_metrics.get("loss", 0.0)),
            float(val_metrics.get("auc", 0.0)),
            float(val_metrics.get("f1", 0.0)),
        )
        
        # Save checkpoint
        safe_val_metrics = _metrics_to_python_floats(val_metrics)
        checkpoint = {
            'epoch': int(epoch + 1),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_auc': float(best_val_auc),
            'val_metrics': safe_val_metrics,
        }
        
        # Save best model
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = float(val_metrics['auc'])
            torch.save(checkpoint, os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth'))
            patience_counter = 0
            log.info("Saved best model (AUC: %.4f)", float(best_val_auc))
        else:
            patience_counter += 1
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(config['paths']['checkpoint_dir'], 'latest.pth'))
        log.info("Saved latest checkpoint")
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping']['patience']:
            log.info("Early stopping triggered at epoch %d", epoch + 1)
            break

        epoch_secs = time.time() - epoch_start
        log.info("Epoch %d finished in %.1fs (ended at %s)", epoch + 1, epoch_secs, _now())
    
    writer.close()
    run_secs = time.time() - run_start
    log.info("Training completed in %.1fs (ended at %s)", run_secs, _now())


if __name__ == "__main__":
    main()

