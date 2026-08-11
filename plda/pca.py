# PATH: plda/pca.py
# Numpy implementation of Principal Component Analysis (PCA)

import numpy as np

class PCA:
    def __init__(self,
                 reg: float = 0.1):
        """
        Initialize the PCA model.

        :param reg: Regularization coefficient for the covariance matrix estimation.
                    covariance_reg = covariance + reg * I
        """

        self.reg = reg

    def train(self,
              X: np.ndarray,
              pca_dim: int = None,
              pca_var: float = None):
        """
        Train the PCA projection matrix for dimensionality reduction.

        :param X: Feature matrix with shape [n_samples, n_features].
                  Each row represents one sample, and each column represents one feature dimension.
        :param pca_dim: Fixed number of the output PCA dimension.
        :param pca_var: Retained explained variance ratio.
                        For example: 0.95 keeps lower-order components to preserve 95% of total variance.

        :return mean: Global mean vector.
        :return V: PCA projection matrix.
        :return dim_out: Output PCA dimension.
        :return explained_var: Actual retained variance ratio.
        """

        X = np.asarray(X, dtype=np.float64)
        n_samples, dim = X.shape

        # 1. Center data
        mean = X.mean(axis=0)
        X = X - mean

        # 2. Calculate total covariance matrix
        covariance = (
            X.T @ X
        ) / n_samples

        # regularization
        covariance = (
                covariance +
                self.reg * np.eye(dim)
        )

        covariance = 0.5 * (covariance + covariance.T)

        # 3. Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(covariance)

        # sort eigenvalues descending
        idx = np.argsort(eigvals)[::-1]

        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # 4. Determine the output PCA dimension
        if pca_dim is not None:
            dim_out = pca_dim

        else:
            explained_variance = (
                np.cumsum(eigvals) /
                np.sum(eigvals)
            )
            dim_out = (np.searchsorted(explained_variance, pca_var) + 1)

        # 5. Calculate actual retained variance
        explained_var = (
                np.sum(eigvals[:dim_out]) /
                np.sum(eigvals)
        )

        # 6. Get the projection matrix
        V = eigvecs[:, :dim_out]

        return mean, V, dim_out, explained_var

    def apply(self,
              data: np.ndarray,
              mean: np.ndarray,
              V: np.ndarray):
        """
        Transform the data by the pre-trained PCA transformation matrix.

        :param data: Feature matrix with shape [n_samples, n_features].
        :param mean: Global mean.
        :param V: PCA transformation matrix returned by PCA.train() with shape [n_features, dim_out].

        :return transformed_data: PCA-transformed data with shape [n_samples, dim_out].
        """

        data = np.asarray(data, dtype=float)
        mean = np.asarray(mean, dtype=float)
        V = np.asarray(V, dtype=float)

        centered_data = data - mean
        transformed_data = centered_data @ V

        return transformed_data

