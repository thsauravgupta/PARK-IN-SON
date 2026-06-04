# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import umap
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.utils import setup_logging, load_config

class ReportGenerator:
    def __init__(self, config):
        self.config = config
        self.logger = setup_logging(__name__)
        self.results_dir = project_root / config["paths"]["results"]
        self.figures_dir = project_root / config["paths"]["figures"]
        self.models_dir = project_root / config["paths"]["models"]
        self.emb_dir = project_root / config["paths"]["embeddings"]
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        (self.figures_dir / "confusion_matrices").mkdir(exist_ok=True)
        (self.figures_dir / "bland_altman_plots").mkdir(exist_ok=True)

    def _format_metrics_table(self):
        csv_path = self.results_dir / "baseline_comparison.csv"
        if not csv_path.exists():
            return
        df = pd.read_csv(csv_path)
        pivot = df.pivot_table(index='model', columns=['task', 'metric_name'], values=['mean', 'std'])
        self.logger.info("Formatted metrics table generated.")
        pivot.to_csv(self.results_dir / "formatted_metrics_table.csv")
        
    def _plot_umap(self):
        try:
            X = pd.read_parquet(self.emb_dir / "fused_embeddings.parquet")
            y_cls = pd.read_parquet(self.emb_dir / "labels.parquet").squeeze()
            
            reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
            embedding = reducer.fit_transform(X)
            
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=y_cls, palette="viridis", s=50, alpha=0.8)
            plt.title("UMAP of Fused Embeddings", fontsize=16)
            plt.tight_layout()
            plt.savefig(self.figures_dir / "umap_fused_embeddings.png", dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"UMAP plot failed: {e}")

    def generate_all(self):
        self._format_metrics_table()
        self._plot_umap()
        self.logger.info("Reports generated successfully.")

if __name__ == '__main__':
    rep = ReportGenerator(load_config())
    rep.generate_all()
