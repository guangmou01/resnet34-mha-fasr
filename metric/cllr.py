# PATH: metric/cllr.py

import numpy as np

from .pav import pav, opt_loglr

def cllr(ss_llr: np.ndarray,
         ds_llr: np.ndarray) -> float:
    """
    log-likelihood-ratio Cost for a given LR system.

    :param ss_llr: same-source LR outputs of natural-log-scale, with shape (n_ss, 1).
    :param ds_llr: different-source LR outputs of natural-log-scale, with shape (n_ds, 1).

    :return cllr: Cllr value as a float.
    """

    ss_llr = np.asarray(ss_llr, dtype=float).ravel()
    ds_llr = np.asarray(ds_llr, dtype=float).ravel()

    punish_ss = np.logaddexp(0, -ss_llr) / np.log(2)
    punish_ds = np.logaddexp(0, ds_llr) / np.log(2)

    n_vali_ss = len(ss_llr)
    n_vali_ds = len(ds_llr)

    cllr_value = 0.5 * (
            1 / n_vali_ss * np.sum(punish_ss) +
            1 / n_vali_ds * np.sum(punish_ds)
    )

    return cllr_value

# Discrimination loss
def cllr_min(ss_llr: np.ndarray,
             ds_llr: np.ndarray) -> float:
    """
    Discrimination loss/ Minimum Cllr value for a given LR system.

    :param ss_llr: same-source LR outputs of natural-log-scale, with shape (n_ss, 1).
    :param ds_llr: different-source LR outputs of natural-log-scale, with shape (n_ds, 1).

    :return cllr_min: Minimum Cllr value as a float.
    """

    ss_llr = np.asarray(ss_llr, dtype=float).ravel()
    ds_llr = np.asarray(ds_llr, dtype=float).ravel()

    opt_res = opt_loglr(ss_llr, ds_llr, option="raw")

    tar_llrs = opt_res["tar_llrs"]
    nontar_llrs = opt_res["nontar_llrs"]

    cllr_min_value = cllr(tar_llrs, nontar_llrs)

    return cllr_min_value

# Calibration loss
def cllr_cal(ss_llr: np.ndarray,
             ds_llr: np.ndarray) -> float:
    """
    Calibration loss for a given LR system.

    :param ss_llr: same-source LR outputs of natural-log-scale, with shape (n_ss, 1).
    :param ds_llr: different-source LR outputs of natural-log-scale, with shape (n_ds, 1).

    :return cllr_cal: Calibration loss value as a float.
    """

    cllr_value = cllr(ss_llr, ds_llr)
    cllr_min_value = cllr_min(ss_llr, ds_llr)

    cllr_cal_value = cllr_value - cllr_min_value

    return cllr_cal_value