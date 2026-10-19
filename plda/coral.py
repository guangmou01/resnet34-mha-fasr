# PATH: plda/coral.py
# Numpy implementation of the CORrelation ALingnment (CORAL) + by-domain mean alignment

import numpy as np

class CORAL:
    def __init__(self):
        """
        Initialize the CORAL model.
        """
        pass

    def sqrtm(self,
              M: np.ndarray,
              eps: float = 1e-12) -> np.ndarray:
        """
        Matrix square root for a symmetric positive semi-definite matrix.

        :param M: A square symmetric matrix.
        :param eps: Minimum eigenvalue floor for numerical stability.

        :return: The matrix square root of M.
        """

        M = 0.5 * (M + M.T)
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, eps)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    def invsqrtm(self,
                 M: np.ndarray,
                 eps: float = 1e-12) -> np.ndarray:
        """
        Matrix square root for a symmetric positive semi-definite matrix.

        :param M: A square symmetric matrix.
        :param eps: Minimum eigenvalue floor for numerical stability.

        :return: The inverse matrix square root of M.
        """

        M = 0.5 * (M + M.T)
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, eps)
        return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    def train(self,
              Xo: np.ndarray,
              Xt: np.ndarray,
              floor: float = 0.5,
              reg: float = 0.1,
              mean_alignment: bool = True,
              eps: float = 1e-12):
        """
        Training module of the CORAL++ (Li et al., 2022) and optional by-domain mean alignment.
        The CORAL++ algorithm is described in:

        Li, R., Zhang, W., & Chen, D. (2022).
        The Coral++ Algorithm for Unsupervised Domain Adaptation of Speaker Recognition.
        2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 7172–7176.
        https://doi.org/10.1109/ICASSP43922.2022.9747792

        :param Xo: Out-of-domain data, with shape [n_samples, n_features].
        :param Xt: Target-domain data, with shape [n_samples, n_features].
        :param floor: Flooring threshold applied after Z-score normalization of target covariance eigenvalues.
        :param reg: Regularization coefficient added to covariance diagonals.
        :param mean_alignment: Whether apply by-domain mean alignment.
        :param eps: Numerical stability floor used in inverse square root and standard deviation checks.

        :return: A list containing the trained parameters of CORAL and optional by-domain mean alignment.
        """

        Xo = np.asarray(Xo, dtype=float)
        Xt = np.asarray(Xt, dtype=float)

        dim = Xo.shape[1]

        if Xo.ndim != 2 or Xt.ndim != 2:
            raise ValueError("Xo and Xt must both be 2D arrays.")
        if Xo.shape[1] != Xt.shape[1]:
            raise ValueError("Xo and Xt must have the same number of features.")
        if Xo.shape[0] < 2 or Xt.shape[0] < 2:
            raise ValueError("Xo and Xt must each contain at least 2 samples.")
        if floor < 0:
            raise ValueError("floor must be non-negative.")
        if reg <= 0:
            raise ValueError("reg must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        # Mean
        mu_o = np.mean(Xo, axis=0)
        mu_t = np.mean(Xt, axis=0)

        # Centering
        Xo_c = Xo - mu_o
        Xt_c = Xt - mu_t

        # Calculate covariance matrix
        cov_o = np.cov(Xo_c, rowvar=False)
        cov_t = np.cov(Xt_c, rowvar=False)

        # Symmetrize to reduce numerical drift
        cov_o = 0.5 * (cov_o + cov_o.T)
        cov_t = 0.5 * (cov_t + cov_t.T)

        # EVD of target covariance
        eigvals_t, eigvecs_t = np.linalg.eigh(cov_t)
        eigvals_t = np.maximum(eigvals_t, 0.0)

        # Z-score normalize eigenvalues
        mean_s = np.mean(eigvals_t)
        std_s = np.std(eigvals_t)
        std_s = max(std_s, eps)
        eigvals_t_z = (eigvals_t - mean_s) / std_s

        # Flooring on normalized eigenvalues
        eigvals_t_floored = np.maximum(floor, eigvals_t_z)

        # Reconstruct CORAL++ target covariance
        cov_o_hat = cov_o + reg * np.eye(dim)
        cov_t_hat = eigvecs_t @ np.diag(eigvals_t_floored) @ eigvecs_t.T
        cov_t_hat = cov_t_hat + reg * np.eye(dim)

        # Get the CORAL++ transformation
        cov_o_inv_sqrt = self.invsqrtm(cov_o_hat, eps=eps)      # Whitening transformation
        cov_t_sqrt = self.sqrtm(cov_t_hat, eps=eps)             # Recoloring transformation
        A = cov_o_inv_sqrt @ cov_t_sqrt                         # CORAL transformation

        CORAL_transformation = [
            ("CORAL_A", A),
            ("CORAL_mu_o", mu_o),
            ("CORAL_mu_t", mu_t),
            ("CORAL_mean_alignment", mean_alignment),
            ("CORAL_dim", dim)
        ]
        return CORAL_transformation

    def apply(self,
              CORAL_transformation,
              data: np.ndarray) -> np.ndarray:
        """
        Align out-of-domain data to the target-domain distribution using CORAL++ (Li et al., 2022) and optional by-domain mean alignment.

        :param CORAL_transformation: The list outputed by CORAL.train().
        :param data: Input out-of-domain data, with shape [n_samples, n_features].

        :return data_aligned: Aligned data, with shape [n_samples, n_features].
        """

        data = np.asarray(data, dtype=float)

        CORAL_params = dict(CORAL_transformation)

        A = CORAL_params["CORAL_A"]
        mu_o = CORAL_params["CORAL_mu_o"]
        mu_t = CORAL_params["CORAL_mu_t"]
        mean_alignment = CORAL_params["CORAL_mean_alignment"]
        dim = CORAL_params["CORAL_dim"]

        if data.ndim != 2:
            raise ValueError("data must be a 2D array.")

        if data.shape[1] != dim:
            raise ValueError("data must have the same number of features as the training data.")

        if mean_alignment:
            data_aligned = (data - mu_o) @ A + mu_t
        else:
            data_aligned = data @ A

        return data_aligned


