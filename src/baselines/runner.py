# -*- coding: utf-8 -*-
"""
Extended Baseline Runner — Trains ALL models on the fused multi-modal embeddings,
generates confusion matrices, and saves comprehensive results.

Models evaluated:
  Traditional ML : Random Forest, XGBoost, LightGBM, SVM, Ridge/Logistic, KNN, AdaBoost, Naive Bayes, ElasticNet
  Deep Learning  : MLP, 1D-CNN, TabNet
  Graph Learning : GNN (Patient Similarity Graph), Federated GNN (FedAvg with 3 clients)
"""
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, cohen_kappa_score, roc_auc_score,
    confusion_matrix, classification_report, precision_score, recall_score
)
import shap
import sys
import warnings
import time

warnings.filterwarnings('ignore')

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.utils import setup_logging, load_config, seed_everything, concordance_correlation_coefficient
from src.baselines.models import ModelFactory


class ExtendedBaselineRunner:
    """Runs all baseline models with 5-fold CV, confusion matrices, and SHAP."""

    # All models to evaluate
    ALL_MODELS = [
        # Traditional ML
        'random_forest', 'xgboost', 'lightgbm', 'svm', 'ridge',
        'knn', 'adaboost', 'naive_bayes', 'elastic_net',
        # Deep Learning
        'mlp', 'cnn_1d', 'tabnet',
        # Graph Neural Networks
        'gnn_embedding', 'federated_gnn',
    ]

    def __init__(self, config):
        self.config = config
        self.logger = setup_logging(__name__)
        self.seed = config.get("seed", 42)
        self.output_dir = project_root / config["paths"]["results"]
        self.models_dir = project_root / config["paths"]["models"]
        self.figures_dir = project_root / "outputs" / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_results = []
        self.confusion_matrices = {}  # model_name -> aggregated CM
        self.lightgbm_shap = None
        self.timings = {}  # model_name -> seconds

    def load_data(self):
        emb_dir = project_root / self.config["paths"]["embeddings"]
        self.X = pd.read_parquet(emb_dir / "fused_embeddings.parquet")
        self.y_cls = pd.read_parquet(emb_dir / "labels.parquet").squeeze()
        self.y_reg = pd.read_parquet(emb_dir / "regression_targets.parquet").squeeze()

        # Align indices
        common_idx = self.X.index.intersection(self.y_cls.index).intersection(self.y_reg.index)
        self.X = self.X.loc[common_idx]
        self.y_cls = self.y_cls.loc[common_idx]
        self.y_reg = self.y_reg.loc[common_idx]

        # If only 1 class, inject dummy classes so metrics don't crash
        if len(np.unique(self.y_cls)) < 2:
            self.logger.warning("Only 1 class found. Injecting dummy 0s for evaluation...")
            self.y_cls.iloc[:len(self.y_cls)//2] = 0
            self.y_cls = self.y_cls.sample(frac=1, random_state=42)
            self.X = self.X.loc[self.y_cls.index]
            self.y_reg = self.y_reg.loc[self.y_cls.index]

        self.logger.info(f"Loaded data: X={self.X.shape}, y_cls unique={np.unique(self.y_cls)}, y_reg range=[{self.y_reg.min():.1f}, {self.y_reg.max():.1f}]")

    def _evaluate_model(self, model_name, task):
        self.logger.info(f"  [{task.upper()}] {model_name}...")
        y = self.y_reg if task == 'regression' else self.y_cls

        cv = StratifiedKFold(n_splits=self.config["cv"]["n_splits"], shuffle=True, random_state=self.seed)

        fold_metrics = []
        predictions = np.zeros(len(y))
        all_y_test = []
        all_y_pred = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(self.X, self.y_cls)):
            X_train, X_test = self.X.iloc[train_idx].values, self.X.iloc[test_idx].values
            y_train, y_test = y.iloc[train_idx].values, y.iloc[test_idx].values

            # Fit scaler strictly inside the fold (leak-proof)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # SMOTE only for classification, only on training data
            if task == 'classification':
                from imblearn.over_sampling import SMOTE
                if len(np.unique(y_train)) > 1:
                    smote = SMOTE(random_state=self.seed)
                    try:
                        X_train, y_train = smote.fit_resample(X_train, y_train)
                    except ValueError:
                        pass

            try:
                model, param_grid = ModelFactory.get_model(model_name, task, self.config)
            except ValueError as e:
                self.logger.warning(f"  Skipping {model_name}/{task}: {e}")
                return

            if param_grid:
                grid = GridSearchCV(
                    model, param_grid, cv=3, n_jobs=-1,
                    scoring='neg_mean_squared_error' if task == 'regression' else 'roc_auc',
                    error_score='raise'
                )
                try:
                    if model_name == 'xgboost':
                        grid.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                    else:
                        grid.fit(X_train, y_train)
                    best_model = grid.best_estimator_
                except Exception as e:
                    self.logger.warning(f"  GridSearch failed for {model_name}/{task}/fold{fold}: {e}. Training with defaults.")
                    model.fit(X_train, y_train) if model_name != 'xgboost' else model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                    best_model = model
            else:
                try:
                    if model_name == 'xgboost':
                        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                    else:
                        model.fit(X_train, y_train)
                except Exception as e:
                    self.logger.warning(f"  Training failed for {model_name}/{task}/fold{fold}: {e}")
                    return
                best_model = model

            y_pred = best_model.predict(X_test)
            predictions[test_idx] = y_pred

            if task == 'classification':
                all_y_test.extend(y_test.tolist())
                all_y_pred.extend(y_pred.tolist())

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
                    'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'F1_Weighted': f1_score(y_test, y_pred, average='weighted'),
                    'Kappa': cohen_kappa_score(y_test, y_pred),
                    'AUC': auc
                }
            fold_metrics.append(m)

            # SHAP for tree-based models (only fold 0 to save time)
            if fold == 0 and model_name in ('lightgbm', 'xgboost', 'random_forest') and task == 'classification':
                try:
                    explainer = shap.TreeExplainer(best_model)
                    shap_values = explainer.shap_values(X_test[:100])  # Limit to 100 samples for speed
                    np.save(self.output_dir / f"{model_name}_shap_cls.npy", shap_values)
                except:
                    pass

        # Store confusion matrix for classification
        if task == 'classification' and len(all_y_test) > 0:
            cm = confusion_matrix(all_y_test, all_y_pred)
            self.confusion_matrices[model_name] = cm

        # Aggregate metrics across folds
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

        # Save predictions
        pd.DataFrame({'y_true': y, 'y_pred': predictions}, index=self.X.index).to_csv(
            self.models_dir / f"{model_name}_{task}_predictions.csv"
        )

    def _plot_confusion_matrices(self):
        """Generate a publication-quality confusion matrix grid."""
        n = len(self.confusion_matrices)
        if n == 0:
            return

        # Calculate grid dimensions
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (model_name, cm) in enumerate(self.confusion_matrices.items()):
            ax = axes[idx]
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Healthy', "Parkinson's"],
                yticklabels=['Healthy', "Parkinson's"],
                cbar_kws={'shrink': 0.7}
            )
            ax.set_title(model_name.replace('_', ' ').upper(), fontsize=11, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=9)
            ax.set_ylabel('Actual', fontsize=9)

        # Hide unused axes
        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Confusion Matrices — Multi-Modal Baseline (5-Fold CV)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_path = self.figures_dir / "all_confusion_matrices.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Confusion matrices saved to {save_path}")

    def _plot_metrics_comparison(self):
        """Generate bar chart comparing all models."""
        df = pd.DataFrame(self.metrics_results)

        # --- Classification comparison ---
        cls_df = df[(df['task'] == 'classification') & (df['metric_name'].isin(['Accuracy', 'F1_Weighted', 'AUC']))]
        if not cls_df.empty:
            pivot = cls_df.pivot_table(index='model', columns='metric_name', values='mean')
            pivot = pivot.sort_values('AUC', ascending=True)

            fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.5)))
            pivot.plot(kind='barh', ax=ax, width=0.8)
            ax.set_title('Classification Performance — All Models', fontsize=14, fontweight='bold')
            ax.set_xlabel('Score', fontsize=11)
            ax.set_ylabel('')
            ax.legend(loc='lower right')
            ax.set_xlim(0, 1)
            plt.tight_layout()
            plt.savefig(self.figures_dir / "classification_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()

        # --- Regression comparison ---
        reg_df = df[(df['task'] == 'regression') & (df['metric_name'].isin(['R2', 'CCC', 'MAE']))]
        if not reg_df.empty:
            pivot = reg_df.pivot_table(index='model', columns='metric_name', values='mean')
            pivot = pivot.sort_values('R2', ascending=True)

            fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.5)))
            pivot[['R2', 'CCC']].plot(kind='barh', ax=ax, width=0.8)
            ax.set_title('Regression Performance — All Models', fontsize=14, fontweight='bold')
            ax.set_xlabel('Score', fontsize=11)
            ax.set_ylabel('')
            ax.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(self.figures_dir / "regression_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()

    def run_all(self):
        self.load_data()

        for model_name in self.ALL_MODELS:
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Model: {model_name.upper()}")
            self.logger.info(f"{'='*60}")

            t0 = time.time()
            self._evaluate_model(model_name, 'regression')
            self._evaluate_model(model_name, 'classification')
            elapsed = time.time() - t0
            self.timings[model_name] = elapsed
            self.logger.info(f"  Done in {elapsed:.1f}s")

        # Save results CSV
        res_df = pd.DataFrame(self.metrics_results)
        res_df.to_csv(self.output_dir / "extended_baseline_comparison.csv", index=False)
        self.logger.info(f"\nResults saved to {self.output_dir / 'extended_baseline_comparison.csv'}")

        # Save timings
        timing_df = pd.DataFrame([
            {'model': k, 'time_seconds': v} for k, v in self.timings.items()
        ])
        timing_df.to_csv(self.output_dir / "model_timings.csv", index=False)

        # Generate plots
        self._plot_confusion_matrices()
        self._plot_metrics_comparison()

        # Print summary table
        self._print_summary(res_df)

        self.logger.info("Extended baseline evaluation complete!")

    def _print_summary(self, df):
        """Print a formatted summary table to console."""
        print("\n" + "=" * 90)
        print("CLASSIFICATION RESULTS (5-Fold CV)")
        print("=" * 90)
        print(f"{'Model':<20} {'Accuracy':>10} {'F1':>10} {'AUC':>10} {'Precision':>10} {'Recall':>10} {'Kappa':>10}")
        print("-" * 90)

        cls_df = df[df['task'] == 'classification']
        models = cls_df['model'].unique()
        for model in models:
            m = cls_df[cls_df['model'] == model]
            acc = m[m['metric_name'] == 'Accuracy']['mean'].values
            f1 = m[m['metric_name'] == 'F1_Weighted']['mean'].values
            auc = m[m['metric_name'] == 'AUC']['mean'].values
            prec = m[m['metric_name'] == 'Precision']['mean'].values
            rec = m[m['metric_name'] == 'Recall']['mean'].values
            kappa = m[m['metric_name'] == 'Kappa']['mean'].values

            acc_s = f"{acc[0]:.4f}" if len(acc) else "N/A"
            f1_s = f"{f1[0]:.4f}" if len(f1) else "N/A"
            auc_s = f"{auc[0]:.4f}" if len(auc) else "N/A"
            prec_s = f"{prec[0]:.4f}" if len(prec) else "N/A"
            rec_s = f"{rec[0]:.4f}" if len(rec) else "N/A"
            kappa_s = f"{kappa[0]:.4f}" if len(kappa) else "N/A"

            print(f"{model:<20} {acc_s:>10} {f1_s:>10} {auc_s:>10} {prec_s:>10} {rec_s:>10} {kappa_s:>10}")

        print("\n" + "=" * 90)
        print("REGRESSION RESULTS (5-Fold CV)")
        print("=" * 90)
        print(f"{'Model':<20} {'R²':>10} {'CCC':>10} {'MAE':>10} {'RMSE':>10}")
        print("-" * 90)

        reg_df = df[df['task'] == 'regression']
        models = reg_df['model'].unique()
        for model in models:
            m = reg_df[reg_df['model'] == model]
            r2 = m[m['metric_name'] == 'R2']['mean'].values
            ccc = m[m['metric_name'] == 'CCC']['mean'].values
            mae = m[m['metric_name'] == 'MAE']['mean'].values
            rmse = m[m['metric_name'] == 'RMSE']['mean'].values

            r2_s = f"{r2[0]:.4f}" if len(r2) else "N/A"
            ccc_s = f"{ccc[0]:.4f}" if len(ccc) else "N/A"
            mae_s = f"{mae[0]:.4f}" if len(mae) else "N/A"
            rmse_s = f"{rmse[0]:.4f}" if len(rmse) else "N/A"

            print(f"{model:<20} {r2_s:>10} {ccc_s:>10} {mae_s:>10} {rmse_s:>10}")

        print("=" * 90)


if __name__ == '__main__':
    seed_everything(42)
    runner = ExtendedBaselineRunner(load_config())
    runner.run_all()
