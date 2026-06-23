# PATH: calibration/loocv_tool.py

import numpy as np
import pandas as pd
from tqdm import tqdm

from .train_BiGauss_regularized import train_BiGauss_regularized
from .BiGauss_calibrator import BiGauss_calibrator
from .lin_fusion import lin_fusion

def BiGauss_LOOCV(csv_path: str,
                  out_csv_path: str,
                  id1: str,
                  id2: str,
                  score: str | list[str],
                  score_scale: str = "ln(LR)",
                  prior: float = 0.5,
                  kappa: float = 0.01,
                  df_reg: int | None = None,
                  max_iter: int = 50000,
                  grid_k: float = 4,
                  grid_len: int = 10000,
                  z_score: bool = True,
                  show_progress: bool = True) -> pd.DataFrame:
    """
    An experiment helper to run the leave-one/two-source-out Bi-Gaussianized calibration.

    :param csv_path: Input .csv file.
    :param out_csv_path: Output .csv file.
    :param id1: Column name of the first source ID.
    :param id2: Column name of the second source ID.
    :param score: Score column name (single-system calibration) or list of score column names (fusion).
    :param score_scale: "ln(LR)", "log10(LR)", or "Raw".
    :param prior: Prior probability for logistic regression training.
    :param kappa: Regularization factor.
    :param df_reg: Pseudo degree of freedom for regularization.
    :param max_iter: Maximum iteration number.
    :param grid_k: BiGauss grid range.
    :param grid_len: BiGauss grid length.
    :param z_score: Whether to apply fold-wise z-score normalization.
    :param show_progress: Whether to show progress bar.

    :return: Output dataframe.
    """

    df = pd.read_csv(csv_path)

    if isinstance(score, str):
        score = [score]

    id1_chr = df[id1].astype(str).to_numpy()
    id2_chr = df[id2].astype(str).to_numpy()

    labels = np.where(id1_chr == id2_chr, "ss", "ds")

    leave_out_key = []
    for a, b in zip(id1_chr, id2_chr):
        x, y = sorted([a, b])
        leave_out_key.append(f"{x}|{y}")

    df["leave_out_key"] = leave_out_key

    scores = df[score].astype(float).to_numpy()

    if score_scale == "Raw":
        scores = np.log(scores)
    elif score_scale == "log10(LR)":
        scores = scores * np.log(10)
    elif score_scale == "ln(LR)":
        pass
    else:
        raise ValueError("score_scale must be 'ln(LR)', 'log10(LR)', or 'Raw'.")

    n = len(df)
    quasi_score = np.full(n, np.nan)
    cllr_target = np.full(n, np.nan)
    sigma2_target = np.full(n, np.nan)
    calibrated_lnLR = np.full(n, np.nan)

    all_ids = sorted(set(id1_chr) | set(id2_chr))
    id2rows = {
        s: np.where((id1_chr == s) | (id2_chr == s))[0]
        for s in all_ids
    }

    keys = pd.unique(df["leave_out_key"])
    iterator = tqdm(keys, desc=f"calibrating {csv_path}") if show_progress else keys

    for key in iterator:
        a, b = key.split("|")

        if a == b:
            excl_idx = id2rows[a]
        else:
            excl_idx = np.union1d(id2rows[a], id2rows[b])

        train_idx = np.setdiff1d(np.arange(n), excl_idx)
        test_idx = np.where(df["leave_out_key"].to_numpy() == key)[0]

        train_scores = scores[train_idx, :]
        test_scores = scores[test_idx, :]

        if z_score:
            mu = train_scores.mean(axis=0)
            sd = train_scores.std(axis=0)
            sd[sd == 0] = 1.0

            train_scores = (train_scores - mu) / sd
            test_scores = (test_scores - mu) / sd

        train_ss = train_scores[labels[train_idx] == "ss", :]
        train_ds = train_scores[labels[train_idx] == "ds", :]

        model = train_BiGauss_regularized(
            targets=train_ss.T,
            non_targets=train_ds.T,
            prior=prior,
            kappa=kappa,
            df=df_reg,
            max_iter=max_iter
        )

        fusion_w = model[0]

        quasi = lin_fusion(
            weights=fusion_w,
            scores=test_scores.T
        )

        cal = BiGauss_calibrator(
            uncal_score=test_scores.T,
            model=model,
            grid_k=grid_k,
            grid_len=grid_len
        )

        quasi_score[test_idx] = np.asarray(quasi).reshape(-1)
        cllr_target[test_idx] = model[1]
        sigma2_target[test_idx] = model[2]
        calibrated_lnLR[test_idx] = np.asarray(cal).reshape(-1)

    df["quasi_score"] = quasi_score
    df["cllr_target"] = cllr_target
    df["sigma2_target"] = sigma2_target
    df["calibrated_lnLR"] = calibrated_lnLR

    df.to_csv(out_csv_path, index=False)

    return df