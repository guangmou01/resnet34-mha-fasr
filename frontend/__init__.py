# PATH: frontend/__init__.py

"""
Frontend feature engineering and deep speaker representation modeling module for pre-trained ResNet34-MHA.

This package contains a full frontend pipeline for:

Waveform -> resampling to 8k mono -> pre-emphasis
         -> FTT to power spectrum -> Fbank -> mean- and variance-normalized Fbank
         -> ResNet34-MHA to 512-dim DNN-speaker-embedding
"""

from .fbank import get_fbank
from .embedding_extractor import EmbeddingExtractor

__version__ = "1.0.0"
__author__ = "Deng, Guangmou"

__all__ = [
    "get_fbank",
    "EmbeddingExtractor"
]