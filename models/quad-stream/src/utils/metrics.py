"""Evaluation metrics for deepfake audio detection.

This project predicts **audio segment-level** scores (e.g., VAD segments) and can
optionally aggregate those scores to **audio file / clip-level** metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from typing import Dict


def compute_segment_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """
    Compute segment-level evaluation metrics.
    
    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_proba: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # ROC-AUC
    try:
        metrics['auc'] = roc_auc_score(y_true, y_proba)
    except ValueError:
        metrics['auc'] = 0.0
    
    # EER (Equal Error Rate)
    metrics['eer'] = compute_eer(y_true, y_proba)
    
    return metrics


def compute_eer(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER).
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        
    Returns:
        EER value
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        fnr = 1 - tpr
        # Find threshold where FPR = FNR
        _ = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
        eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
        return float(eer)
    except ValueError:
        # Happens when only one class is present in y_true.
        return 0.0


def compute_file_metrics(
    file_predictions: Dict[str, np.ndarray],
    file_labels: Dict[str, int],
    aggregation: str = "mean",
) -> Dict[str, float]:
    """
    Compute audio file / clip-level metrics by aggregating segment predictions.
    
    Args:
        file_predictions: Dictionary mapping sample_id to array/list of segment probabilities
        file_labels: Dictionary mapping sample_id to label
        aggregation: Aggregation method ('mean', 'max', 'median')
        
    Returns:
        Dictionary of file-level metrics
    """
    file_probas = []
    file_labels_list = []
    
    for sample_id, predictions in file_predictions.items():
        if sample_id not in file_labels:
            continue
        
        # Aggregate segment predictions
        if aggregation == "mean":
            file_proba = np.mean(predictions)
        elif aggregation == "max":
            file_proba = np.max(predictions)
        elif aggregation == "median":
            file_proba = np.median(predictions)
        else:
            file_proba = np.mean(predictions)
        
        file_probas.append(file_proba)
        file_labels_list.append(file_labels[sample_id])
    
    file_probas = np.array(file_probas)
    file_labels_list = np.array(file_labels_list)
    file_preds = (file_probas > 0.5).astype(int)
    
    return compute_segment_metrics(file_labels_list, file_preds, file_probas)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred)


