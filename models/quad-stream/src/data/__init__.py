"""Data loading utilities.

This project originally targeted video frame datasets, but it now includes an
audio dataset pipeline that produces 4 frequency-image inputs per sample:

- segment STFT
- segment log-mel
- full audio STFT
- full audio log-mel
"""

from .dataset import DeepfakeDataset  # backwards-compatible import

