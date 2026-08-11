# PATH: plda/lda.py
# Numpy + Scipy implementation of Regularized Linear Discriminant Function (LDF/LDA)

import numpy as np
from scipy.linalg import eigh

class LDA:
    def __init__(self,
                 shrinkage: float = 0.1,
                 eps: float = 1e-12):
        """
        Initialize the LDA model.

        :param shrinkage: Shrinkage coefficient for Sw.
                          Sw_reg = (1 - shrinkage) * Sw + shrinkage * scale_Sw * I
        :param eps: Eigenvalue floor used after regularizing Sw.
                    Eigenvalues are floored at eps.
        """

        self.shrinkage = shrinkage
        self.eps = eps

    def train(self,
              y: np.ndarray,
              X: np.ndarray,
              lda_dim: int) -> np.ndarray:
        """
        Train the LDA transformation matrix for dimensionality reduction.

        Hastie, T., Tibshirani, R., & Friedman, J. (2009).
        The Elements of Statistical Learning: Data Mining, Inference, and Prediction (2nd ed.). Springer.
        https://doi.org/10.1007/978-0-387-84858-7

        :param y: Class labels for the input samples, which should be a column array with shape [n_samples, 1].
        :param X: Feature matrix with shape [n_samples, n_features].
                  Each row represents one sample, and each column represents one feature dimension.
        :param lda_dim: Target dimensionality after LDA transformation.
                        It mustn't exceed min(n_classes - 1, n_features).

        :return V: LDA transformation matrix V. Each column is one discriminant direction.
                   The transformed features can be obtained by feature @ V.
        """

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).reshape(-1)

        n_sample, dim = X.shape
        classes = np.unique(y)
        n_class = len(classes)

        if lda_dim > min(n_class - 1, dim):
            raise ValueError(
                f"lda_dim = {lda_dim} is not valid. "
                f"It must be <= {min(n_class - 1, dim)}."
            )

        # 1. Calculate global mean
        mean_global = X.mean(axis=0)

        # 2. Calculate within-class and between-class scatter
        Sw = np.zeros((dim, dim), dtype=np.float64)
        Sb = np.zeros((dim, dim), dtype=np.float64)

        for cls in classes:
            Xc = X[y == cls]
            mean_class = Xc.mean(axis=0)

            Xc_centered = Xc - mean_class
            Sw += Xc_centered.T @ Xc_centered

            mean_diff = (mean_class - mean_global).reshape(1, -1)
            Sb += Xc.shape[0] * (mean_diff.T @ mean_diff)

        # Force symmetry
        Sw = 0.5 * (Sw + Sw.T)
        Sb = 0.5 * (Sb + Sb.T)

        # Scale-aware shrinkage regularization of Sw
        scale_Sw = np.trace(Sw) / dim if np.trace(Sw) > 0 else 1.0
        Sw_reg = (1 - self.shrinkage) * Sw + self.shrinkage * scale_Sw * np.eye(dim)
        Sw_reg = 0.5 * (Sw_reg + Sw_reg.T)
        Sw_reg += self.eps * np.eye(dim)

        # 3. Solve generalized eigenvalue problem:
        # Sb * V = lambda * Sw_reg * V
        eigvals, eigvecs = eigh(Sb, Sw_reg)
        descending_idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, descending_idx]
        V = eigvecs[:, :lda_dim]

        return V

    def apply(self,
              data: np.ndarray,
              V: np.ndarray) -> np.ndarray:
        """
        Transform the data by the pre-trained LDA transformation matrix.

        :param data: Feature matrix with shape [n_samples, n_features].
        :param V: LDA transformation matrix returned by LDA.train() with shape [n_features, lda_dim].

        :return transformed_data: LDA-transformed data with shape [n_samples, lda_dim].
        """

        data = np.asarray(data, dtype=float)
        V = np.asarray(V, dtype=float)

        transformed_data = data @ V

        return transformed_data




