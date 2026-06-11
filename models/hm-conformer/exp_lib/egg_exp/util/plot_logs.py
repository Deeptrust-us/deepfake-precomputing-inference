"""
Utility script to parse and plot training logs from LocalLogger.

The logs are stored as .txt files with the following format:
- With step: [step] metric_name: value
- Without step: metric_name: value
"""

import os
import re
import argparse
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
from pathlib import Path


def parse_log_file(log_path):
    """
    Parse a log file and extract metrics.
    
    Args:
        log_path: Path to the log .txt file
        
    Returns:
        steps: List of step values (or None if no steps)
        values: List of metric values
        has_steps: Boolean indicating if the log has step information
    """
    if not os.path.exists(log_path):
        return None, None, False
    
    steps = []
    values = []
    has_steps = False
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Try to match format with step: [step] metric_name: value
            match_with_step = re.match(r'\[(\d+)\]\s*\w+:\s*([\d.eE+-]+)', line)
            if match_with_step:
                step = int(match_with_step.group(1))
                value = float(match_with_step.group(2))
                steps.append(step)
                values.append(value)
                has_steps = True
            else:
                # Try to match format without step: metric_name: value
                match_no_step = re.match(r'\w+:\s*([\d.eE+-]+)', line)
                if match_no_step:
                    value = float(match_no_step.group(1))
                    values.append(value)
                    has_steps = False
    
    if has_steps:
        return steps, values, True
    else:
        return list(range(len(values))), values, False


def plot_metrics(log_dir, metrics=None, save_path=None, show=True):
    """
    Plot metrics from log files.
    
    Args:
        log_dir: Directory containing log files
        metrics: List of metric names to plot (if None, plots all found metrics)
        save_path: Path to save the plot (if None, doesn't save)
        show: Whether to display the plot
    """
    log_dir = Path(log_dir)
    
    if plt is None:
        print("Matplotlib not installed. Skipping plot.")
        return

    # Find all .txt log files (excluding parameters.txt and description.txt)
    log_files = list(log_dir.glob('*.txt'))
    log_files = [f for f in log_files if f.stem not in ['parameters', 'description']]
    
    if not log_files:
        print(f"No log files found in {log_dir}")
        return
    
    # Filter by requested metrics
    if metrics:
        log_files = [f for f in log_files if f.stem in metrics]
    
    if not log_files:
        print(f"No matching log files found for metrics: {metrics}")
        return
    
    # Determine subplot layout
    n_metrics = len(log_files)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, log_file in enumerate(log_files):
        metric_name = log_file.stem
        steps, values, has_steps = parse_log_file(log_file)
        
        if steps is None or values is None:
            print(f"Warning: Could not parse {log_file}")
            continue
        
        ax = axes[idx] if n_metrics > 1 else axes[0]
        
        if has_steps:
            ax.plot(steps, values, marker='o', markersize=3, linewidth=1.5)
            ax.set_xlabel('Epoch/Step', fontsize=12)
        else:
            ax.plot(values, marker='o', markersize=3, linewidth=1.5)
            ax.set_xlabel('Iteration', fontsize=12)
        
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        if len(values) > 0:
            mean_val = np.mean(values)
            min_val = np.min(values)
            max_val = np.max(values)
            stats_text = f'Mean: {mean_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_metric_comparison(log_dirs, metric_name, labels=None, save_path=None, show=True):
    """
    Compare a single metric across multiple log directories.
    
    Args:
        log_dirs: List of log directories to compare
        metric_name: Name of the metric to compare
        labels: List of labels for each directory (if None, uses directory names)
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if plt is None:
        print("Matplotlib not installed. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    if labels is None:
        labels = [Path(d).name for d in log_dirs]
    
    for log_dir, label in zip(log_dirs, labels):
        log_path = Path(log_dir) / f'{metric_name}.txt'
        steps, values, has_steps = parse_log_file(log_path)
        
        if steps is None or values is None:
            print(f"Warning: Could not parse {log_path}")
            continue
        
        if has_steps:
            ax.plot(steps, values, marker='o', markersize=3, linewidth=1.5, label=label)
        else:
            ax.plot(values, marker='o', markersize=3, linewidth=1.5, label=label)
    
    ax.set_xlabel('Epoch/Step' if has_steps else 'Iteration', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def list_available_metrics(log_dir):
    """List all available metrics in a log directory."""
    log_dir = Path(log_dir)
    log_files = list(log_dir.glob('*.txt'))
    metrics = [f.stem for f in log_files if f.stem not in ['parameters', 'description']]
    return sorted(metrics)


def main():
    parser = argparse.ArgumentParser(description='Plot training logs from LocalLogger')
    parser.add_argument('log_dir', type=str, help='Path to log directory (e.g., /results/ASVspoof2023/HM-Conformer)')
    parser.add_argument('--metrics', type=str, nargs='+', default=None,
                       help='Specific metrics to plot (default: all)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save the plot (default: don\'t save)')
    parser.add_argument('--no-show', action='store_true',
                       help='Don\'t display the plot')
    parser.add_argument('--list', action='store_true',
                       help='List available metrics and exit')
    
    args = parser.parse_args()
    
    if args.list:
        metrics = list_available_metrics(args.log_dir)
        print(f"Available metrics in {args.log_dir}:")
        for metric in metrics:
            print(f"  - {metric}")
        return
    
    plot_metrics(args.log_dir, args.metrics, args.save, show=not args.no_show)


if __name__ == '__main__':
    main()



