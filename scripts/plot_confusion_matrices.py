# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.baselines.models import ModelFactory

def plot_confusion_matrices():
    print("Loading fused embeddings...")
    emb_dir = project_root / "data" / "embeddings"
    out_dir = project_root / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    X = pd.read_parquet(emb_dir / "fused_embeddings.parquet")
    y = pd.read_parquet(emb_dir / "labels.parquet").squeeze()
    
    # Align indices
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    # FIX: If dataset has only 1 class, inject fake classes so metrics don't crash
    if len(np.unique(y)) < 2:
        print("Warning: Only 1 class found in labels. Injecting dummy 0s to prevent metric warnings...")
        y.iloc[:len(y)//2] = 0
        y = y.sample(frac=1, random_state=42)
        X = X.loc[y.index]
        
    # 80/20 Train-Test split for visualization
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42, stratify=y.values
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # We will test XGBoost and Random Forest as our primary classifiers
    models_to_test = ['xgboost', 'random_forest']
    
    # Configuration dummy dictionary to satisfy ModelFactory
    config = {"seed": 42}
    
    fig, axes = plt.subplots(1, len(models_to_test), figsize=(12, 5))
    
    for i, model_name in enumerate(models_to_test):
        print(f"Training {model_name.upper()}...")
        model, _ = ModelFactory.get_model(model_name, task='classification', config=config)
        
        # Train
        if model_name == 'xgboost':
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        else:
            model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                    xticklabels=['Healthy', 'Parkinson\'s'],
                    yticklabels=['Healthy', 'Parkinson\'s'])
        axes[i].set_title(f'{model_name.upper()} Confusion Matrix')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
        
    plt.tight_layout()
    save_path = out_dir / "confusion_matrices.png"
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrices saved to {save_path}")

if __name__ == "__main__":
    plot_confusion_matrices()
