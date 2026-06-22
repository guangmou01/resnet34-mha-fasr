# PATH: plda/two_cov_plda.py

import numpy as np

def sort_svd(s, d):
    idx = np.argsort(-s)
    s1 = s[idx]
    d1 = d.T
    d1 = d1[idx].T

    return s1, d1

class TwoCovPLDA:

    def __init__(self):
        pass

    def train(self,
              y: np.ndarray,
              X: np.ndarray,
              n_iter: int = 50) -> dict:

        """
        Train the Two-Covariance PLDA using EM algorithm (Sizov et al., 2014).

        Sizov, A., Lee, K. A., & Kinnunen, T. (2014).
        Unifying Probabilistic Linear Discriminant Analysis Variants in Biometric Authentication.
        In P. Fränti, G. Brown, M. Loog, F. Escolano, & M. Pelillo (Eds.),
        Structural, Syntactic, and Statistical Pattern Recognition (pp. 464–475). Springer.
        https://doi.org/10.1007/978-3-662-44415-3_47

        :param y: Class labels with shape [n_samples,].
        :param X: Feature matrix with shape [n_samples, n_features].
        :param n_iter: Number of EM iterations.

        :return model: Dictionary containing trained PLDA parameters.
        """

        y = np.asarray(y).reshape(-1)
        X = np.asarray(X, dtype=np.float64)
        N, dim = X.shape    # the zero-order moment 'N', (B.1)

        unique_labels = np.unique(y)
        K = len(unique_labels)

        # the first-order moment for the i-th source
        f_list = []
        n_list = []

        for lab in unique_labels:
            Xi = X[y == lab]
            n_list.append(Xi.shape[0])
            f_list.append(np.sum(Xi, axis=0))

        n_list = np.asarray(n_list, dtype=np.int64)
        f_list = np.asarray(f_list, dtype=np.float64)   # the first-order moment 'f_i', (B.2)

        # the global second-order moment 'S', (B.3)
        S = X.T @ X

        # initialization
        mu = np.zeros(dim)
        B_cov = np.eye(dim)
        W_cov = np.eye(dim)

        B_prec = np.linalg.inv(B_cov)
        W_prec = np.linalg.inv(W_cov)

        # EM iterations
        for it in range(n_iter):

            T = np.zeros((dim, dim), dtype=np.float64)
            R = np.zeros((dim, dim), dtype=np.float64)
            Y = np.zeros(dim, dtype=np.float64)             # set: T <- 0, R <- 0, Y <- 0

            # E-step
            for n_i, f_i in zip(n_list, f_list):

                L_i = B_prec + n_i * W_prec                 # (B.4)

                gamma_i = B_prec @ mu + W_prec @ f_i
                E_y_i = np.linalg.solve(L_i, gamma_i)       # (B.5)

                L_i_inv = np.linalg.inv(L_i)
                E_yy_i = L_i_inv + np.outer(E_y_i, E_y_i)   # (B.6)

                T += np.outer(E_y_i, f_i)                   # (B.8)
                R += n_i * E_yy_i                           # (B.9)
                Y += n_i * E_y_i                            # (B.10)

            # M-step
            mu = Y / N                                                                          # (B.11)
            B_cov = (R - np.outer(mu, Y.transpose()) - np.outer(Y, mu)) / N + np.outer(mu, mu)  # (B.12)
            W_cov = (S - T - T.transpose() + R) / N                                             # (B.13)

            # Force symmetry
            B_cov = 0.5 * (B_cov + B_cov.T)
            W_cov = 0.5 * (W_cov + W_cov.T)

            B_prec = np.linalg.inv(B_cov)
            W_prec = np.linalg.inv(W_cov)

        # get PLDA transformation for closed-form scoring (Kaldi-style PLDA parameterization)
        # transform the vector into the space, where the W_cov is unit 'I', and the B_cov is diagonalized
        # adapt from Wespeaker toolkit:
        # https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/utils/plda/two_cov_plda.py
        try:
            transform1 = np.linalg.cholesky(W_cov)
        except np.linalg.LinAlgError:
            transform1 = np.linalg.cholesky(W_cov + np.eye(W_cov.shape[0]) * 1e-12)

        transform1 = np.linalg.inv(transform1)

        B_proj = transform1 @ B_cov @ transform1.T
        B_proj = 0.5 * (B_proj + B_proj.T)

        psi, U = np.linalg.eigh(B_proj)

        psi = np.where(psi > 0.0, psi, 0.0)
        psi, U = sort_svd(psi, U)

        plda_transform = U.T @ transform1

        offset = -1.0 * (plda_transform @ mu)

        model = {
            # Metadata
            "dim": np.array(dim),
            "n_samples": np.array(N),
            "n_classes": np.array(K),
            "n_iter": np.array(n_iter),
            # Mean
            "mu": mu,
            # Precision matrices
            "B_prec": B_prec,
            "W_prec": W_prec,
            # Covariance matrices
            "B_cov": B_cov,
            "W_cov": W_cov,
            # PLDA transformation for closed-form scoring
            "transform": plda_transform,
            "psi": psi,
            "offset": offset
        }

        return model

    def scoring(self,
                x_q: np.ndarray,
                x_k: np.ndarray,
                model: dict) -> float:
        """
        Calculate common-source LR by pre-trained PLDA model.

        :param x_q: Questioned-source vector with shape [1, n_features].
        :param x_k: Known-source vector with shape [1, n_features].
        :param model: The dictionary returned by TwoCovPLDA.train().

        :return score: Uncalibrated natural-log-likelihood-ratio, ln(LR).
        """

        x_q = np.asarray(x_q, dtype=np.float64)
        x_k = np.asarray(x_k, dtype=np.float64)

        transform = np.asarray(model["transform"], dtype=np.float64)
        offset = np.asarray(model["offset"], dtype=np.float64)
        psi = np.asarray(model["psi"], dtype=np.float64)

        x_q = x_q @ transform.T + offset
        x_k = x_k @ transform.T + offset

        x_q = x_q.reshape(-1)
        x_k = x_k.reshape(-1)

        psi = np.maximum(psi, 0.0)

        # Same-source hypothesis
        det_same = 1.0 + 2.0 * psi

        quad_same = (
                ((1.0 + psi) * (x_q ** 2 + x_k ** 2) - 2.0 * psi * x_q * x_k)
                / det_same
        )

        loglike_same = -0.5 * (
                np.sum(np.log(det_same)) +
                np.sum(quad_same)
        )

        # Different-source hypothesis
        det_diff = (1.0 + psi) ** 2

        quad_diff = (x_q ** 2 + x_k ** 2) / (1.0 + psi)

        loglike_diff = -0.5 * (
                np.sum(np.log(det_diff)) +
                np.sum(quad_diff)
        )

        score = loglike_same - loglike_diff

        return score

