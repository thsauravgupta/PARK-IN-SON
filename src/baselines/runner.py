import numpy as np
from sklearn.model_selection import KFold
import logging
from src.baselines.models import get_baseline_models
from src.evaluation.metrics import concordance_correlation_coefficient
from src.utils import setup_logging

def run_baselines(features, targets, n_splits=5):
    """
    Evaluates baseline models on flat feature matrices using K-Fold CV.
    Target metric is CCC.
    features: np.ndarray shape (N, total_features)
    targets: np.ndarray shape (N)
    """
    logger = setup_logging(__name__)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = get_baseline_models()
    
    results = {m: [] for m in models.keys()}
    
    logger.info(f"Running baselines with {n_splits}-fold CV on shape {features.shape}...")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(features)):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            ccc = concordance_correlation_coefficient(y_test, preds)
            results[name].append(ccc)
            
    logger.info("Baseline Validation Results (CCC):")
    for name, scores in results.items():
        logger.info(f"{name}: {np.mean(scores):.4f} \u00b1 {np.std(scores):.4f}")
        
    return results

if __name__ == "__main__":
    # Test script with random data if run independently
    X = np.random.randn(200, 150)
    y = np.random.randn(200) * 10 + X[:, 0]*5
    run_baselines(X, y)
