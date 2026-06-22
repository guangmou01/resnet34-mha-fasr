# PATH: plda/__init__.py

"""
PLDA backend for LR-based forensic comparison task

This package contains a full backend pipeline for:
Feature data -> LDF/LDA -> centering -> whitening -> length transformation -> two-covariance PLDA
"""

from .lda import LDA
from .pipeline import pipeline
from .two_cov_plda import TwoCovPLDA
from .transformation import Whiten, RG, LN

__version__ = "1.0.0"
__author__ = "Deng, Guangmou"

__all__ = [
    "LDA",
    "Whiten",
    "RG",
    "LN",
    "pipeline",
    "TwoCovPLDA"
]