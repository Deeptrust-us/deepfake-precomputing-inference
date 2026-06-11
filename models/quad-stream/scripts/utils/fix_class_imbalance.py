"""Fix class imbalance issue in training."""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import WeightedRandomSampler
import json
import os

def get_class_weights(metadata_file):
    """Calculate class weights for balanced training."""
    with open(metadata_file, 'r') as f:
        data = json.load(f)
    
    labels = [item['label'] for item in data]
    class_counts = np.bincount(labels)
    
    # Calculate weights: inverse frequency
    total = len(labels)
    num_classes = len(class_counts)
    weights = total / (num_classes * class_counts)
    
    # Normalize
    weights = weights / weights.sum() * num_classes
    
    print(f"Class counts: {dict(zip(range(num_classes), class_counts))}")
    print(f"Class weights: {dict(zip(range(num_classes), weights))}")
    
    return weights, labels

def create_weighted_sampler(metadata_file):
    """Create weighted sampler for balanced batches."""
    weights, labels = get_class_weights(metadata_file)
    
    # Create sample weights
    sample_weights = [weights[label] for label in labels]
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

if __name__ == "__main__":
    # Check current class distribution
    print("=== Current Class Distribution ===\n")
    
    for split in ['train', 'val', 'test']:
        meta_file = f'data/{split}_metadata.json'
        if os.path.exists(meta_file):
            weights, labels = get_class_weights(meta_file)
            print()



