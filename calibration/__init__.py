# PATH: calibration/__init__.py

"""
Score-to-LR calibration package for LR-based forensic comparison task.
"""

from .train_llr_fusion_regularized import train_llr_fusion_regularized
from .lin_fusion import lin_fusion
from .train_BiGauss_regularized import train_BiGauss_regularized
from .BiGauss_calibrator import BiGauss_calibrator
from .loocv_tool import BiGauss_LOOCV

__version__ = "1.0.0"
__author__ = "Deng, Guangmou"

__all__ = [
    "train_llr_fusion_regularized",
    "lin_fusion",
    "train_BiGauss_regularized",
    "BiGauss_calibrator",
    "BiGauss_LOOCV"
]