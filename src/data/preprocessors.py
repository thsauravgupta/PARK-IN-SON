import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class BasePreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X_train: pd.DataFrame):
        raise NotImplementedError
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
        
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)

class ClinicalPreprocessor(BasePreprocessor):
    def fit(self, X_train: pd.DataFrame):
        self.imputer = SimpleImputer(strategy='mean')
        self.cols = X_train.select_dtypes(include=[np.number]).columns
        if len(self.cols) > 0:
            self.scaler.fit(X_train[self.cols])
            self.imputer.fit(self.scaler.transform(X_train[self.cols]))
        self.is_fitted = True
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if len(self.cols) > 0:
            out[self.cols] = self.scaler.transform(out[self.cols])
            out[self.cols] = self.imputer.transform(out[self.cols])
        return out.fillna(0)

class MRIPreprocessor(BasePreprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.n_rois = config.get("mri", {}).get("n_rois", 100)
        
    def fit(self, X_train: pd.DataFrame):
        self.is_fitted = True
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Create pure zero arrays where missing MRI will lead to missing_mask=1 in dataset
        patnos = X.index
        # Simulating random MRI features for the sake of the base pipeline, 
        # actual missing individuals will be explicitly zeroed in higher layers
        np.random.seed(42)
        mri = np.random.randn(len(patnos), self.n_rois)
        # Randomly blank out 20% to simulate missing modalities
        mask = np.random.rand(len(patnos)) < 0.2
        mri[mask] = 0.0
        
        df = pd.DataFrame(mri, index=patnos, columns=[f"ROI_{i}" for i in range(self.n_rois)])
        return df

class PETPreprocessor(BasePreprocessor):
    def fit(self, X_train: pd.DataFrame):
        self.cols = ["caudate_mean", "putamen_mean", "asymmetry_caudate"]
        self.is_fitted = True
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        patnos = X.index
        np.random.seed(43)
        pet = np.random.randn(len(patnos), len(self.cols))
        # Zero out 15% missing
        mask = np.random.rand(len(patnos)) < 0.15
        pet[mask] = 0.0
        return pd.DataFrame(pet, index=patnos, columns=self.cols)

class GeneticPreprocessor(BasePreprocessor):
    def fit(self, X_train: pd.DataFrame):
        self.is_fitted = True
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        patnos = X.index
        np.random.seed(44)
        genetics = np.random.randint(0, 3, size=(len(patnos), 6))
        # Zero out missing
        mask = np.random.rand(len(patnos)) < 0.10
        genetics = genetics.astype('float32')
        genetics[mask] = 0.0
        return pd.DataFrame(genetics, index=patnos, columns=["LRRK2", "GBA", "SNCA", "PINK1", "PRKN", "APOE"])
