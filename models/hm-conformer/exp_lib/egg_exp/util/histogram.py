import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import torch

def histogram(scores, labels, title='histogram', name1='True', name2='False'):  # only T, F / numpy_arr
    if plt is None:
        return None
    fig = plt.figure(figsize=(8,6))
    
    mask_T = (labels == 0)
    mask_F = (labels == 1)
    plt.hist(scores[mask_T], bins=100, alpha=0.5, label=name1)
    plt.hist(scores[mask_F], bins=100, alpha=0.5, label=name2)
    
    plt.xlabel("Score", size=14)
    plt.ylabel("Count", size=14)
    plt.title(title)
    plt.legend(loc='upper right')
    return fig
