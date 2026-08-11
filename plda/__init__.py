# PATH: plda/__init__.py

"""
PLDA backend for LR-based forensic comparison task

This package contains a full backend pipeline for:
Feature data -> dimensionality reduction -> centering -> whitening -> length transformation -> two-covariance PLDA
"""

from .lda import LDA
from .pca import PCA
from .transformation import Whiten, RG, LN
from .two_cov_plda import TwoCovPLDA
from .pipeline_lda import pipeline_lda
from .pipeline_pca import pipeline_pca

__version__ = "1.0.0"
__author__ = "Deng, Guangmou"

__all__ = [
    "LDA",
    "PCA",
    "Whiten",
    "RG",
    "LN",
    "TwoCovPLDA",
    "pipeline_lda",
    "pipeline_pca"
]