# PATH: calibration/BiGauss_calibrator.py

import numpy as np
from scipy.interpolate import interp1d

from .lin_fusion import lin_fusion

# example input scores array:
#          s1,  s2, ...,  sn
# sys-1: [0.8, 1.0, ..., 0.9],
# sys-2: [1.5, 1.7, ..., 1.7],
# sys-3: [0.3, 1.4, ..., 0.8],
# ...
# sys-d: [0.6, 0.7, ..., 2.4].

def BiGauss_calibrator(uncal_score: float | np.ndarray,
                       model: list,
                       grid_k: float = 4,
                       grid_len: int = 10000) -> np.ndarray:
    """
    Calibrate scores by a pretrained Bi-Gaussianized calibration model.

    reference:
    Morrison, G. S. (2024).
    Bi-Gaussianized calibration of likelihood ratios.
    Law, Probability and Risk, 23(1), 1–34.
    https://doi.org/10.1093/lpr/mgae004

    :param uncal_score: [d, n], uncalibrated score(s).
    :param model: BiGauss model returned by train_BiGauss().
    :param grid_k: Range (in multiples of sigma) for grid construction.
    :param grid_len: Number of grid points.

    :return calibrated_lnLR: [n, 1] array of the well-calibrated ln(LR).
    """

    fusion_w, _, sigma2_target, weighted_ecdf, bigmm_cdf = model

    X = np.asarray(uncal_score, dtype=float)

    if X.ndim == 0:
        X = X.reshape(1, 1)
    elif X.ndim == 1:
        X = X.reshape(1, -1)

    d = len(fusion_w) - 1
    if X.shape[0] != d:
        raise ValueError("Mismatch in system dimension (uncalibrated score).")

    # Step-1: pre-calibrate/fuse the score by LogReg
    quasi_uncal = lin_fusion(weights=fusion_w, scores=X)
    quasi_uncal = np.asarray(quasi_uncal, dtype=float)

    # Step-2: search the ECDF value of the score in the calibration set
    qvals = weighted_ecdf(quasi_uncal)

    # Step-3: map the ECDF value to ln(LR) in the perfectly-calibrated BiGauss distribution
    half_sigma2 = sigma2_target / 2.0
    sigma_target = np.sqrt(sigma2_target)

    lnLR_max = half_sigma2 + grid_k * sigma_target
    lnLR_min = -lnLR_max

    grid = np.linspace(lnLR_min, lnLR_max, grid_len)
    target_cdf = bigmm_cdf(grid)

    # remove duplicated CDF values (required for interpolation)
    cdf_unique, idx = np.unique(target_cdf, return_index=True)
    grid_unique = grid[idx]

    inv_cdf = interp1d(
        cdf_unique,
        grid_unique,
        bounds_error=False,
        fill_value=(grid_unique[0], grid_unique[-1])
    )

    calibrated_lnLR = np.asarray(inv_cdf(qvals), dtype=float)
    return calibrated_lnLR.reshape(-1, 1)

