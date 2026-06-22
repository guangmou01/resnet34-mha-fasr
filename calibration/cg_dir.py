# PATH: calibration/cg_dir.py

import numpy as np

def cg_dir(old_dir: np.ndarray,
           grad: np.ndarray,
           old_grad: np.ndarray) -> np.ndarray:
    """
    Compute the updating conjugate-gradient search direction, used in 'train_llr_fusion_regularized.py' module.

    :param old_dir: previous search direction, with shape [d+1, 1].
    :param grad: current gradient, with shape [d+1, 1].
    :param old_grad: previous gradient, with shape [d+1, 1].

    :return dir: new search direction, with shape [d+1, 1].
    """

    g = grad
    grad = grad.flatten()

    old_grad = old_grad.flatten()

    delta = grad - old_grad

    den = old_dir.T.dot(delta)
    if den == 0:
        dir = g*0
    else:
        beta = np.dot(grad, delta) / den
        dir = g - beta * old_dir

    return dir