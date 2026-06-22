# PATH: metric/pav.py

import numpy as np

# Pool Adjacent Violators
def pav(y: np.ndarray) -> np.ndarray:
    """
    Apply the Pool Adjacent Violators algorithm.

    :param y: input sequence to be monotonically fitted, with shape [n,].

    :return ghat: monotonic fitted sequence, with shape [n,].
    """

    y = np.asarray(y, dtype=float)
    n = len(y)

    index = np.zeros(n, dtype=int)
    length = np.zeros(n, dtype=int)
    ghat = np.zeros(n, dtype=float)

    ci = 0

    index[ci] = 0
    length[ci] = 1
    ghat[ci] = y[0]

    if n >= 2:
        for j in range(1, n):
            ci += 1
            index[ci] = j
            length[ci] = 1
            ghat[ci] = y[j]

            while ci >= 1 and ghat[ci - 1] >= ghat[ci]:
                nw = length[ci - 1] + length[ci]
                ghat[ci - 1] = (
                    ghat[ci - 1]
                    + (length[ci] / nw) * (ghat[ci] - ghat[ci - 1])
                )
                length[ci - 1] = nw
                ci -= 1

    # backward expansion
    n_tmp = n
    while n_tmp >= 1:
        for j in range(index[ci], n_tmp):
            ghat[j] = ghat[ci]
        n_tmp = index[ci]
        ci -= 1

    return ghat

# Non-parametric optimal monotonic log-LR mapping
def opt_loglr(tar_scores: np.ndarray,
              nontar_scores: np.ndarray,
              option="laplace") -> dict:
    """
    Estimate optimal monotonic log-likelihood-ratio mappings using PAV.

    :param tar_scores: target / same-source scores, with shape (Nt,).
    :param nontar_scores: non-target / different-source scores, with shape (Nn,).
    :param option: smoothing option; use "laplace" to apply Laplace-style padding.

    :return: dictionary containing target and non-target log-LRs.
    """

    tar_scores = np.asarray(tar_scores, dtype=float)
    nontar_scores = np.asarray(nontar_scores, dtype=float)

    Nt = len(tar_scores)
    Nn = len(nontar_scores)
    N = Nt + Nn

    tar_scores = tar_scores - 1.0e-6

    scores = np.concatenate([nontar_scores, tar_scores])

    Pideal = np.concatenate([np.zeros(Nn), np.ones(Nt)])

    ord_idx = np.argsort(scores)
    scores = scores[ord_idx]
    Pideal = Pideal[ord_idx]

    if option == "laplace":
        Pideal = np.concatenate([[1, 0], Pideal, [1, 0]])

    Popt = pav(Pideal)

    if option == "laplace":
        Popt = Popt[2:-2]

    with np.errstate(divide="ignore"):
        posterior_log_odds = np.log(Popt) - np.log(1.0 - Popt)

    log_prior_odds = np.log(Nt) - np.log(Nn)

    llrs = posterior_log_odds - log_prior_odds

    llrs = llrs + (np.arange(1, N + 1) * 1.0e-6 / N)

    llrs_unsorted = np.zeros(N)
    llrs_unsorted[ord_idx] = llrs

    nontar_llrs = llrs_unsorted[:Nn]
    tar_llrs = llrs_unsorted[Nn:Nn + Nt]

    return {
        "tar_llrs": tar_llrs,
        "nontar_llrs": nontar_llrs
    }
