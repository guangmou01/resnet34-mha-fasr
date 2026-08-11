# PATH: plda/pipeline_lda.py
# A wrapped PCA-PLDA pipeline for backend LR-score calculation

# This pipeline includes transformation:
# Feature data -> PCA -> centering -> whitening -> length transformation -> two-covariance PLDA

import os
import h5py
import numpy as np

from .pca import PCA
from .transformation import Whiten, RG, LN
from .two_cov_plda import TwoCovPLDA

class pipeline_pca:

    def __init__(self):
        pass

    def train(self,
              training_set_path: str = "training_set.h5",
              label_key: str = "labels",
              feature_key: str = "embeddings",
              pca_dim: int = None,
              pca_var: float = None,
              pca_reg: float = 0.1,
              whiten_reg: float = 0.1,
              whiten_eps: float = 1e-6,
              length_transformation: str = "RG",
              em_iter: int = 50,
              save_path: str = "pipeline.h5") -> None:
        """
        :param training_set_path: Path to the training set (should be a .h5 file).
        :param label_key: Dataset key for labels in the .h5 file.
        :param feature_key: Dataset key for training embeddings/features in the .h5 file.
        :param pca_dim: Fixed number of the output PCA dimension.
        :param pca_var: Retained explained variance ratio.
        :param pca_reg: Regularization coefficient for the covariance matrix estimation.
        :param whiten_reg: Regularization coefficient in Whiten training.
        :param whiten_eps: Eigenvalue floor used in Whiten training.
        :param length_transformation: Transformation method for vector length.
                                      "RG" - Radial Gaussianization
                                      "LN" - Length Normalization/Equalization
        :param em_iter: Number of EM iterations for TwoCovPLDA.
        :param save_path: Path for saving the trained pipeline model/parameters.

        :return: A .h5 file containing all trained pipeline parameters.
        """

        # 1. Load the training data
        print(f"Step 1: Loading training data from {training_set_path}")

        with h5py.File(training_set_path, "r") as f:
            labels = f[label_key][:]
            features = f[feature_key][:]

        labels = np.asarray(labels).reshape(-1)
        print("Number of sources:", len(np.unique(labels)))
        features = np.asarray(features, dtype=np.float64)
        print("Data shape:", features.shape)

        # 2. PCA
        print("Step 2: Training PCA")

        pca = PCA(reg=pca_reg)

        PCA_mean, PCA_V, PCA_dim, PCA_var = pca.train(
            X=features,
            pca_dim=pca_dim,
            pca_var=pca_var
        )

        print(
            f"PCA dimension: {PCA_dim} "
            f"(retained variance: {PCA_var:.4f})"
        )

        features_pca = pca.apply(data=features, mean=PCA_mean, V=PCA_V)

        # 3. Centering
        print("Step 3: Centering")

        Center_mean = np.mean(features_pca, axis=0)
        features_centered = features_pca - Center_mean

        # 4. Whitening
        print("Step 4: Whitening")

        whitener = Whiten(reg=whiten_reg, eps=whiten_eps)
        Whiten_W = whitener.train(X=features_centered)
        features_whiten = whitener.apply(data=features_centered, W=Whiten_W)

        # 5. Length transformation
        print("Step 5: Length transformation")

        rg = RG()
        rg_model = rg.train(X=features_whiten)
        if length_transformation == "RG":
            features_transformed = rg.apply(data=features_whiten, model=rg_model)
        elif length_transformation == "LN":
            features_transformed = LN(X=features_whiten)
        else:
            raise ValueError(f"Unsupported length_transformation type: {length_transformation}")

        # 6. Train TwoCovPLDA
        print("Step 6: Training TwoCovPLDA")

        plda = TwoCovPLDA()
        plda_model = plda.train(y=labels,
                                X=features_transformed,
                                n_iter=em_iter)

        # 7. Save model
        print(f"Step 7: Saving model to {save_path}")

        save_dir = os.path.dirname(save_path)

        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

        with h5py.File(save_path, "w") as f:
            # PCA
            f.create_dataset("PCA_mean",            data=PCA_mean)
            f.create_dataset("PCA_V",               data=PCA_V)

            # Centering
            f.create_dataset("Center_mean",         data=Center_mean)

            # Whitening
            f.create_dataset("Whiten_W",            data=Whiten_W)

            # RG
            f.create_dataset("RG_radius_sorted",    data=rg_model["radius_sorted"])
            f.create_dataset("RG_target_radius",    data=rg_model["target_radius"])
            f.create_dataset("RG_dim",              data=rg_model["dim"])
            f.create_dataset("RG_eps",              data=rg_model["eps"])

            # PLDA
            f.create_dataset("PLDA_mu",             data=plda_model["mu"])
            f.create_dataset("PLDA_B_prec",         data=plda_model["B_prec"])
            f.create_dataset("PLDA_W_prec",         data=plda_model["W_prec"])
            f.create_dataset("PLDA_B_cov",          data=plda_model["B_cov"])
            f.create_dataset("PLDA_W_cov",          data=plda_model["W_cov"])
            f.create_dataset("PLDA_transform",      data=plda_model["transform"])
            f.create_dataset("PLDA_psi",            data=plda_model["psi"])
            f.create_dataset("PLDA_offset",         data=plda_model["offset"])

            # Metadata
            f.create_dataset("N_sources",               data=np.array(len(np.unique(labels))))
            f.create_dataset("N_samples",               data=np.array(features.shape[0]))
            f.create_dataset("Input_dim",               data=np.array(features.shape[1]))
            f.create_dataset("PCA_var",                 data=np.array(PCA_var))
            f.create_dataset("PCA_dim",                 data=np.array(PCA_dim))
            f.create_dataset("PCA_reg",                 data=np.array(pca_reg))
            f.create_dataset("Whiten_reg",              data=np.array(whiten_reg))
            f.create_dataset("Whiten_eps",              data=np.array(whiten_eps))
            f.create_dataset("Length_Transformation",   data=np.bytes_(length_transformation))
            f.create_dataset("EM_iter",                 data=np.array(em_iter))

        print("Model saved")

    def scoring(self,
                x_q: np.ndarray,
                x_k: np.ndarray,
                model_path: str = "pipeline.h5") -> float:
        """
        Calculate common-source LR by a pre-trained pipeline model.

        :param x_q: Questioned-source vector with shape [1, n_features].
        :param x_k: Known-source vector with shape [1, n_features].
        :param model_path: The .h5 file returned by pipeline_pca.train().

        :return score: Uncalibrated natural-log-likelihood-ratio score, ln(LR).
        """

        x_q = np.asarray(x_q, dtype=np.float64)
        x_k = np.asarray(x_k, dtype=np.float64)

        with h5py.File(model_path, "r") as f:
            PCA_mean = f["PCA_mean"][:]
            PCA_V = f["PCA_V"][:]

            Center_mean = f["Center_mean"][:]

            Whiten_W = f["Whiten_W"][:]

            RG_model = {
                "radius_sorted": f["RG_radius_sorted"][:],
                "target_radius": f["RG_target_radius"][:],
                "dim": f["RG_dim"][()],
                "eps": f["RG_eps"][()]
            }

            Length_Transformation = (
                f["Length_Transformation"][()]
                .decode("utf-8")
            )

            PLDA_model = {
                "transform": f["PLDA_transform"][:],
                "offset": f["PLDA_offset"][:],
                "psi": f["PLDA_psi"][:]
            }

        # 1. PCA
        pca = PCA()
        x_q_pca = pca.apply(data=x_q, mean=PCA_mean, V=PCA_V)
        x_k_pca = pca.apply(data=x_k, mean=PCA_mean, V=PCA_V)

        # 2. Centering
        x_q_centered = x_q_pca - Center_mean
        x_k_centered = x_k_pca - Center_mean

        # 3. Whitening
        whitener = Whiten()
        x_q_whiten = whitener.apply(data=x_q_centered, W=Whiten_W)
        x_k_whiten = whitener.apply(data=x_k_centered, W=Whiten_W)

        # 4. Length Transformation
        if Length_Transformation == "RG":
            rg = RG()
            x_q_transformed = rg.apply(data=x_q_whiten, model=RG_model)
            x_k_transformed = rg.apply(data=x_k_whiten, model=RG_model)
        elif Length_Transformation == "LN":
            x_q_transformed = LN(x_q_whiten)
            x_k_transformed = LN(x_k_whiten)
        else:
            raise ValueError(f"Unsupported length_transformation type: {Length_Transformation}")

        # 5. PLDA scoring
        plda = TwoCovPLDA()
        score = plda.scoring(x_q=x_q_transformed, x_k=x_k_transformed, model=PLDA_model)

        return score



