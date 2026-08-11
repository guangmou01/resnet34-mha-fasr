# PATH: plda/transformation.py
# Gaussianization tools for PLDA front-end pre-processing

import numpy as np
from scipy.stats import chi

class Whiten:
    """
    Numpy implementation of regularized whitening.
    """

    def __init__(self,
                 reg: float = 0.1,
                 eps: float = 1e-12):
        """
        Initialize the Whiten model.

        :param reg: Regularization coefficient for Sigma.
                    Sigma_reg = Sigma + reg * I
        :param eps: Eigenvalue floor used after regularizing W.
                    Eigenvalues are floored at eps.
        """

        self.reg = reg
        self.eps = eps

    def train(self,
              X: np.ndarray) -> np.ndarray:

        """
        Train the Whiten transformation matrix W for whitening data.

        :param X: Training data matrix with shape [n_samples, n_features].
                  Each row is one sample and each column is one feature dimension.

        :return W: Whitening transformation matrix W with shape [n_features, n_features].
        """

        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        # 1. Estimate the total covariance matrix.
        Sigma = np.cov(X, rowvar=False, bias=False)

        # Force symmetry.
        Sigma = 0.5 * (Sigma + Sigma.T)

        # 2. Regularize the total covariance matrix.
        # Sigma_reg = Sigma + reg * I
        Sigma_reg = Sigma + self.reg * np.eye(n_features)

        # 3. Eigen-decomposition.
        eigvals, eigvecs = np.linalg.eigh(Sigma_reg)
        eigvals = np.maximum(eigvals, self.eps)

        # 4. Inverse square root:
        # W = Sigma_reg^(-1/2)
        W = (eigvecs
             @ np.diag(1.0 / np.sqrt(eigvals))
             @ eigvecs.T)

        # Force symmetry.
        W = 0.5 * (W + W.T)

        return W

    def apply(self,
              data: np.ndarray,
              W: np.ndarray) -> np.ndarray:
        """
        Apply a whitening transformation matrix to data.

        :param data: Data matrix with shape [n_samples, n_features].
        :param W: Whitening transformation matrix with shape [n_features, n_features].

        :return X_whitened: Whitened data with shape [n_samples, n_features].
        """

        data = np.asarray(data, dtype=float)
        W = np.asarray(W, dtype=float)

        data_whiten = data @ W

        return data_whiten


class RG:
    """
    Numpy + Scipy implementation of Radial Gaussianization (RG).

    Lyu, S., & Simoncelli, E. P. (2009).
    Nonlinear Extraction of Independent Components of Natural Images Using Radial Gaussianization.
    Neural Computation, 21(6), 1485–1519.
    https://doi.org/10.1162/neco.2009.04-08-773
    """

    def __init__(self,
                 eps: float = 1e-12):
        """
        Initialize the Radial Gaussianization model.

        :param eps: Small positive value used for numerical stability.
        """

        self.eps = eps

    def train(self,
              X: np.ndarray) -> dict:
        """
        Train the Radial Gaussianization mapping.

        :param X: Training vectors with shape [n_samples, n_features].

        :return model: A dictionary containing the trained radius mapping.
        """

        X = np.asarray(X, dtype=float)
        n_samples, dim = X.shape

        # 1. Compute training radii.
        radius = np.sqrt(np.sum(X ** 2, axis=1))

        # 2. Sort empirical radii and compute ECDF
        radius_sorted = np.sort(radius)
        prob = (np.arange(n_samples) + 0.5) / n_samples

        # 3. Target chi-distributed radii
        target_radius = chi.ppf(prob, df=dim)

        model = {
            "radius_sorted": radius_sorted,
            "target_radius": target_radius,
            "dim": np.array(dim),
            "eps": np.array(self.eps),
        }

        return model

    def apply(self,
              data: np.ndarray,
              model: dict) -> np.ndarray:
        """
        Apply Radial Gaussianization.

        :param data: Input vectors with shape [n_samples, n_features].
        :param model: The dictionary returned by RG.train().

        :return data_rg: Radial-Gaussianized vectors.
        """

        data = np.asarray(data, dtype=np.float64)

        radius_sorted = model["radius_sorted"]
        target_radius = model["target_radius"]
        eps = float(model["eps"])

        # 1. Compute input radii
        radius = np.sqrt(np.sum(data ** 2, axis=1))

        # 2. Map empirical radius to chi-distributed radius
        new_radius = np.interp(
            radius,
            radius_sorted,
            target_radius,
            left=target_radius[0],
            right=target_radius[-1],
        )

        # 3. Rescale the vector to the new radius
        scale = np.divide(
            new_radius,
            radius,
            out=np.ones_like(radius),
            where=radius > eps
        )
        data_rg = data * scale[:, None]

        return data_rg

def LN(X: np.ndarray,
       eps: float = 1e-12) -> np.ndarray:
    """
    Apply length normalization/equalization to row vectors.

    Garcia-Romero, D., & Espy-Wilson, C. Y. (2011).
    Analysis of i-vector length normalization in speaker recognition systems.
    Proc. Interspeech 2011, 249–252.
    https://doi.org/10.21437/Interspeech.2011-53

    :param X: Data matrix with shape [n_samples, n_features].
    :param eps: Small value used to avoid division by zero.

    :return X_ln: Length-normalized vectors with the same shape as X.
    """

    X = np.asarray(X, dtype=np.float64)

    norm = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
    X_ln = X / np.maximum(norm, eps)

    return X_ln


