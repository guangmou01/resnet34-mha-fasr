# PATH: calibration/lin_fusion.py

import numpy as np

# example input scores array:
#          s1,  s2, ...,  sn
# sys-1: [0.8, 1.0, ..., 0.9],
# sys-2: [1.5, 1.7, ..., 1.7],
# sys-3: [0.3, 1.4, ..., 0.8],
# ...
# sys-d: [0.6, 0.7, ..., 2.4].

def lin_fusion(weights: np.ndarray,
               scores: np.ndarray | float) -> np.ndarray:
    """
    Linear logistic regression calibration/fusion by pre-trained weights, adapted from (Brümmer, 2005).

    reference:
    Brümmer, N. (2005).
    FoCal Toolbox [MATLAB script].
    http://www.dsp.sun.ac.za/nbrummer/focal

    :param weights: d system weights + bias pre-trained by train_llr_fusion(), shape [d+1, 1].
    :param scores: n scores for each of d systems, shape [d, n].

    :return f: Calibrated or fused scores, shape [n,].
    """

    weights = np.asarray(weights).flatten()
    scores = np.asarray(scores)

    d, n = scores.shape

    if len(weights) != d + 1:
        print("Warning: mismatch in system dimension")
        return None

    augmented_scores = np.vstack([scores, np.ones((1, n))])

    f = weights @ augmented_scores
    return f
