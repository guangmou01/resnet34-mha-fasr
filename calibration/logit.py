# PATH: calibration/logit.py

import numpy as np

def logit(p):
    """
    Compute the logit of a probability, used in 'train_llr_fusion_regularized.py' module.

    :param p: probability value in the range (0, 1).

    :return: logit value, log(p / (1 - p)).
    """
    return np.log(p/(1-p))