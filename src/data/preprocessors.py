# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer

# class BasePreprocessor:
#     def __init__(self, config):
#         self.config = config
#         self.scaler = StandardScaler()
#         self.is_fitted = False
        
#     def fit(self, X_train: pd.DataFrame):
#         raise NotImplementedError
        
#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         raise NotImplementedError
        
#     def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         self.fit(X)
#         return self.transform(X)

# class ClinicalPreprocessor(BasePreprocessor):
#     def fit(self, X_train: pd.DataFrame):
#         self.imputer = SimpleImputer(strategy='mean')
#         self.cols = X_train.select_dtypes(include=[np.number]).columns
#         if len(self.cols) > 0:
#             self.scaler.fit(X_train[self.cols])
#             self.imputer.fit(self.scaler.transform(X_train[self.cols]))
#         self.is_fitted = True
#         return self
        
#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         out = X.copy()
#         if len(self.cols) > 0:
#             out[self.cols] = self.scaler.transform(out[self.cols])
#             out[self.cols] = self.imputer.transform(out[self.cols])
#         return out.fillna(0)

# class MRIPreprocessor(BasePreprocessor):
#     def __init__(self, config):
#         super().__init__(config)
#         self.n_rois = config.get("mri", {}).get("n_rois", 100)
        
#     def fit(self, X_train: pd.DataFrame):
#         self.is_fitted = True
#         return self
        
#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         # Create pure zero arrays where missing MRI will lead to missing_mask=1 in dataset
#         patnos = X.index
#         # Simulating random MRI features for the sake of the base pipeline, 
#         # actual missing individuals will be explicitly zeroed in higher layers
#         np.random.seed(42)
#         mri = np.random.randn(len(patnos), self.n_rois)
#         # Randomly blank out 20% to simulate missing modalities
#         mask = np.random.rand(len(patnos)) < 0.2
#         mri[mask] = 0.0
        
#         df = pd.DataFrame(mri, index=patnos, columns=[f"ROI_{i}" for i in range(self.n_rois)])
#         return df

# class PETPreprocessor(BasePreprocessor):
#     def fit(self, X_train: pd.DataFrame):
#         self.cols = ["caudate_mean", "putamen_mean", "asymmetry_caudate"]
#         self.is_fitted = True
#         return self
        
#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         patnos = X.index
#         np.random.seed(43)
#         pet = np.random.randn(len(patnos), len(self.cols))
#         # Zero out 15% missing
#         mask = np.random.rand(len(patnos)) < 0.15
#         pet[mask] = 0.0
#         return pd.DataFrame(pet, index=patnos, columns=self.cols)

# class GeneticPreprocessor(BasePreprocessor):
#     def fit(self, X_train: pd.DataFrame):
#         self.is_fitted = True
#         return self
        
#     def transform(self, X: pd.DataFrame) -> pd.DataFrame:
#         patnos = X.index
#         np.random.seed(44)
#         genetics = np.random.randint(0, 3, size=(len(patnos), 6))
#         # Zero out missing
#         mask = np.random.rand(len(patnos)) < 0.10
#         genetics = genetics.astype('float32')
#         genetics[mask] = 0.0
#         return pd.DataFrame(genetics, index=patnos, columns=["LRRK2", "GBA", "SNCA", "PINK1", "PRKN", "APOE"])

# -*- coding: utf-8 -*-
"""
Legacy modality-specific preprocessors for Fed-PhenoGraft.

These classes are retained for backwards compatibility with the original
preprocessor API and tests.

The production data pipeline uses:
    clinical_loader.py
    mri_pipeline.py
    pet_loader.py
    genetic_loader.py
    preprocessing.py

Unlike the original implementation, these preprocessors do NOT generate
random PET/genetic values. They deterministically derive features from the
data supplied to them.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


class BasePreprocessor:
    def __init__(self, config):
        self.config = config
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame):
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)


class ClinicalPreprocessor(BasePreprocessor):
    """
    Clinical feature preprocessing.

    Numeric clinical features are median-imputed using training data.
    We deliberately do not standardize here because the production
    preprocessing pipeline in preprocessing.py owns train-only scaling.
    """

    def fit(self, X_train: pd.DataFrame):
        self.cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

        self.imputer = None

        if self.cols:
            self.imputer = SimpleImputer(strategy="median")
            self.imputer.fit(X_train[self.cols])

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError(
                "ClinicalPreprocessor must be fitted before transform()."
            )

        out = X.copy()

        if self.cols:
            transformed = self.imputer.transform(out[self.cols])

            out.loc[:, self.cols] = pd.DataFrame(
                transformed,
                index=out.index,
                columns=self.cols,
            )

        return out


class MRIPreprocessor(BasePreprocessor):
    """
    Backwards-compatible MRI preprocessor.

    If real MRI/FreeSurfer features are supplied, they are preserved.
    If only PATNO/index information is supplied, a deterministic zero-filled
    synthetic representation is returned.

    The production project should use mri_pipeline.py for real FreeSurfer
    features and the MRI GNN encoder for representation learning.
    """

    def __init__(self, config):
        super().__init__(config)

        self.n_rois = int(
            config.get("mri", {}).get("n_rois", 100)
        )

    def fit(self, X_train: pd.DataFrame):
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError(
                "MRIPreprocessor must be fitted before transform()."
            )

        # If actual numeric MRI features are supplied, preserve them.
        numeric_cols = X.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        if numeric_cols:
            return X[numeric_cols].copy()

        # Compatibility fallback for the original synthetic-mode test.
        #
        # IMPORTANT:
        # Do not generate random biological measurements. Missing MRI is
        # represented by zeros so that the downstream modality mask can
        # identify it correctly.
        return pd.DataFrame(
            np.zeros((len(X), self.n_rois), dtype=np.float32),
            index=X.index,
            columns=[f"ROI_{i}" for i in range(self.n_rois)],
        )


class PETPreprocessor(BasePreprocessor):
    """
    DaTScan/PET feature engineering.

    Expected canonical raw inputs:
        caudate_r
        caudate_l
        putamen_r
        putamen_l

    Produces the same ten-feature representation used by the real
    pet_loader.py:

        caudate_r
        caudate_l
        putamen_r
        putamen_l
        caudate_mean
        putamen_mean
        asymmetry_caudate
        asymmetry_putamen
        striatum_total
        caudate_putamen_ratio
    """

    def __init__(self, config):
        super().__init__(config)

        self.epsilon = float(
            config.get("pet", {}).get("epsilon", 1e-8)
        )

        self.output_columns = [
            "caudate_r",
            "caudate_l",
            "putamen_r",
            "putamen_l",
            "caudate_mean",
            "putamen_mean",
            "asymmetry_caudate",
            "asymmetry_putamen",
            "striatum_total",
            "caudate_putamen_ratio",
        ]

    def fit(self, X_train: pd.DataFrame):
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError(
                "PETPreprocessor must be fitted before transform()."
            )

        out = pd.DataFrame(index=X.index)

        # Accept common capitalization variants.
        aliases = {
            "caudate_r": [
                "caudate_r",
                "CAUDATE_R",
                "DATSCAN_CAUDATE_R",
            ],
            "caudate_l": [
                "caudate_l",
                "CAUDATE_L",
                "DATSCAN_CAUDATE_L",
            ],
            "putamen_r": [
                "putamen_r",
                "PUTAMEN_R",
                "DATSCAN_PUTAMEN_R",
            ],
            "putamen_l": [
                "putamen_l",
                "PUTAMEN_L",
                "DATSCAN_PUTAMEN_L",
            ],
        }

        for target, candidates in aliases.items():
            source = next(
                (column for column in candidates if column in X.columns),
                None,
            )

            if source is None:
                out[target] = 0.0
            else:
                out[target] = pd.to_numeric(
                    X[source],
                    errors="coerce",
                ).fillna(0.0)

        out["caudate_mean"] = (
            out["caudate_r"] + out["caudate_l"]
        ) / 2.0

        out["putamen_mean"] = (
            out["putamen_r"] + out["putamen_l"]
        ) / 2.0

        out["asymmetry_caudate"] = (
            (out["caudate_r"] - out["caudate_l"]).abs()
            / (out["caudate_mean"] + self.epsilon)
        )

        out["asymmetry_putamen"] = (
            (out["putamen_r"] - out["putamen_l"]).abs()
            / (out["putamen_mean"] + self.epsilon)
        )

        out["striatum_total"] = (
            out["caudate_r"]
            + out["caudate_l"]
            + out["putamen_r"]
            + out["putamen_l"]
        )

        out["caudate_putamen_ratio"] = (
            out["caudate_mean"]
            / (out["putamen_mean"] + self.epsilon)
        )

        return out[self.output_columns]


class GeneticPreprocessor(BasePreprocessor):
    """
    Genetic feature preprocessing.

    Canonical output contains carrier indicators for the six target genes,
    plus derived burden/risk features.

    PARK2 and PRKN are treated as aliases for the same gene.
    """

    def __init__(self, config):
        super().__init__(config)

        configured_genes = config.get(
            "genetic", {}
        ).get(
            "target_genes",
            ["LRRK2", "GBA", "SNCA", "PINK1", "PARK2", "APOE"],
        )

        self.target_genes = list(configured_genes)

    @staticmethod
    def _find_column(
        X: pd.DataFrame,
        candidates: list[str],
    ):
        for column in candidates:
            if column in X.columns:
                return column
        return None

    @staticmethod
    def _carrier_indicator(series: pd.Series) -> pd.Series:
        """
        Convert PPMI-style genetic values to a binary carrier indicator.

        Examples:
            0 / "0" / "negative" -> 0
            1 / "G2019S" / "N409S" -> 1
            NaN / "not tested" -> 0
        """
        numeric = pd.to_numeric(series, errors="coerce")

        result = pd.Series(
            np.zeros(len(series), dtype=np.int64),
            index=series.index,
        )

        numeric_mask = numeric.notna()
        result.loc[numeric_mask] = (
            numeric.loc[numeric_mask] > 0
        ).astype(np.int64)

        text = (
            series.astype(str)
            .str.strip()
            .str.lower()
        )

        negative = text.isin([
            "",
            "nan",
            "none",
            "na",
            "n/a",
            "unknown",
            "not tested",
            "not_tested",
            "not done",
            "not_done",
            "negative",
            "neg",
            "false",
            "no",
            "0",
            "0.0",
        ])

        non_numeric = ~numeric_mask

        result.loc[non_numeric] = (
            (~negative.loc[non_numeric])
            .astype(np.int64)
        )

        return result

    def fit(self, X_train: pd.DataFrame):
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError(
                "GeneticPreprocessor must be fitted before transform()."
            )

        out = pd.DataFrame(index=X.index)

        # Canonical gene aliases.
        gene_aliases = {
            "LRRK2": [
                "LRRK2",
                "LRRK2_MUTATION",
                "LRRK2_STATUS",
            ],
            "GBA": [
                "GBA",
                "GBA_MUTATION",
                "GBA_STATUS",
            ],
            "SNCA": [
                "SNCA",
                "SNCA_MUTATION",
                "SNCA_STATUS",
            ],
            "PINK1": [
                "PINK1",
                "PINK1_MUTATION",
                "PINK1_STATUS",
            ],
            "PRKN": [
                "PRKN",
                "PARK2",
                "PARKIN",
                "PARK2_MUTATION",
            ],
        }

        canonical_genes = [
            "LRRK2",
            "GBA",
            "SNCA",
            "PINK1",
            "PRKN",
        ]

        for gene in canonical_genes:
            source = self._find_column(
                X,
                gene_aliases[gene],
            )

            if source is None:
                out[gene] = 0
            else:
                out[gene] = self._carrier_indicator(
                    X[source]
                )

        # APOE is normally represented as genotype/carrier information.
        apoe_source = self._find_column(
            X,
            [
                "APOE",
                "APOE_GENOTYPE",
                "APOE4_STATUS",
                "APOE_e4_carrier",
            ],
        )

        if apoe_source is None:
            out["APOE_e4_carrier"] = 0
        elif apoe_source == "APOE_e4_carrier":
            out["APOE_e4_carrier"] = self._carrier_indicator(
                X[apoe_source]
            )
        else:
            values = X[apoe_source]

            numeric = pd.to_numeric(
                values,
                errors="coerce",
            )

            result = pd.Series(
                np.zeros(len(values), dtype=np.int64),
                index=values.index,
            )

            numeric_mask = numeric.notna()
            result.loc[numeric_mask] = (
                numeric.loc[numeric_mask] > 0
            ).astype(np.int64)

            text = (
                values.astype(str)
                .str.strip()
                .str.lower()
            )

            result.loc[~numeric_mask] = (
                text.loc[~numeric_mask]
                .str.contains(
                    r"e4|4",
                    regex=True,
                    na=False,
                )
                .astype(np.int64)
            )

            out["APOE_e4_carrier"] = result

        # Number of pathogenic variants/carrier genes.
        carrier_cols = [
            gene
            for gene in canonical_genes
            if gene in out.columns
        ]

        out["n_variants"] = out[carrier_cols].sum(axis=1)

        # Explicit features expected by the original API/tests.
        out["lrrk2_positive"] = (
            out["LRRK2"] > 0
        ).astype(np.int64)

        out["gba_positive"] = (
            out["GBA"] > 0
        ).astype(np.int64)

        return out