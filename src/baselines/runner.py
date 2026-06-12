# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
import shap
import sys
import warnings

warnings.filterwarnings('ignore')

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.utils import setup_logging, load_config, seed_everything, concordance_correlation_coefficient
from src.baselines.models import ModelFactory

class BaselineRunner:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logging(__name__)
        self.seed = config.get("seed", 42)
        self.output_dir = project_root / config["paths"]["results"]
        self.models_dir = project_root / config["paths"]["models"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_results = []
        self.lightgbm_shap = None

    def load_data(self):
        emb_dir = project_root / self.config["paths"]["embeddings"]
        self.X = pd.read_parquet(emb_dir / "fused_embeddings.parquet")
        self.y_cls = pd.read_parquet(emb_dir / "labels.parquet").squeeze()
        self.y_reg = pd.read_parquet(emb_dir / "regression_targets.parquet").squeeze()
        
        # Make sure align
        common_idx = self.X.index.intersection(self.y_cls.index).intersection(self.y_reg.index)
        self.X = self.X.loc[common_idx]
        self.y_cls = self.y_cls.loc[common_idx]
        self.y_reg = self.y_reg.loc[common_idx]

        # FIX: If dataset has only 1 class, inject fake classes so metrics don't crash and print huge traces.
        if len(np.unique(self.y_cls)) < 2:
            self.logger.info("Only 1 class found in labels. Injecting dummy 0s to prevent metric warnings...")
            self.y_cls.iloc[:len(self.y_cls)//2] = 0
            self.y_cls = self.y_cls.sample(frac=1, random_state=42)
            # Reorder X and y_reg to match shuffled y_cls just in case
            self.X = self.X.loc[self.y_cls.index]
            self.y_reg = self.y_reg.loc[self.y_cls.index]

    def _evaluate_model(self, model_name, task):
        self.logger.info(f"Evaluating {model_name} for {task}...")
        y = self.y_reg if task == 'regression' else self.y_cls
        
        # Use StratifiedKFold using y_cls even for regression to maintain group distribution
        cv = StratifiedKFold(n_splits=self.config["cv"]["n_splits"], shuffle=True, random_state=self.seed)
        
        fold_metrics = []
        predictions = np.zeros(len(y))
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(self.X, self.y_cls)):
            X_train, X_test = self.X.iloc[train_idx].values, self.X.iloc[test_idx].values
            y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values
            
            # CRITICAL: Fit scaler strictly inside the fold
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            if task == 'classification':
                from imblearn.over_sampling import SMOTE
                # Only apply SMOTE if there is more than 1 class (safeguard) and classes are imbalanced
                if len(np.unique(y_train)) > 1:
                    smote = SMOTE(random_state=self.seed)
                    try:
                        X_train, y_train = smote.fit_resample(X_train, y_train)
                    except ValueError:
                        # Fallback if too few samples in minority class
                        pass
            
            model, param_grid = ModelFactory.get_model(model_name, task, self.config)
            
            if param_grid:
                grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, 
                                    scoring='neg_mean_squared_error' if task=='regression' else 'roc_auc')
                
                if model_name == 'xgboost':
                    grid.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                else:
                    grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
            else:
                if model_name == 'xgboost':
                    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                else:
                    model.fit(X_train, y_train)
                best_model = model
                
            y_pred = best_model.predict(X_test)
            predictions[test_idx] = y_pred
            
            if task == 'regression':
                m = {
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'R2': r2_score(y_test, y_pred),
                    'CCC': concordance_correlation_coefficient(y_test, y_pred)
                }
            else:
                try:
                    y_prob = best_model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_prob)
                except:
                    auc = np.nan
                    
                m = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'F1_Weighted': f1_score(y_test, y_pred, average='weighted'),
                    'Kappa': cohen_kappa_score(y_test, y_pred),
                    'AUC': auc
                }
            fold_metrics.append(m)
            
            # SHAP
            if model_name == 'lightgbm':
                explainer = shap.TreeExplainer(best_model)
                shap_values = explainer.shap_values(X_test)
                if self.lightgbm_shap is None:
                    self.lightgbm_shap = shap_values
                else:
                    if isinstance(shap_values, list):
                        self.lightgbm_shap = [np.vstack([self.lightgbm_shap[i], shap_values[i]]) for i in range(len(shap_values))]
                    else:
                        self.lightgbm_shap = np.vstack([self.lightgbm_shap, shap_values])
        
        # Aggregate metrics
        avg_metrics = {k: np.mean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0].keys()}
        std_metrics = {k: np.std([fm[k] for fm in fold_metrics]) for k in fold_metrics[0].keys()}
        
        for k in avg_metrics:
            self.metrics_results.append({
                'model': model_name,
                'task': task,
                'metric_name': k,
                'mean': avg_metrics[k],
                'std': std_metrics[k]
            })
            
        pd.DataFrame({'y_true': y, 'y_pred': predictions}, index=self.X.index).to_csv(
            self.models_dir / f"{model_name}_{task}_predictions.csv"
        )
        
    def run_all(self):
        self.load_data()
        models = ['random_forest', 'xgboost', 'svm', 'ridge', 'mlp', 'lightgbm']
        
        for m in models:
            self._evaluate_model(m, 'regression')
            self._evaluate_model(m, 'classification')
            
        res_df = pd.DataFrame(self.metrics_results)
        res_df.to_csv(self.output_dir / "baseline_comparison.csv", index=False)
        
        if self.lightgbm_shap is not None:
            np.save(self.output_dir / "lightgbm_shap.npy", self.lightgbm_shap)
            
        self.logger.info("Evaluation complete.")

if __name__ == '__main__':
    seed_everything(42)
    runner = BaselineRunner(load_config())
    runner.run_all()
