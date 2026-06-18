# -*- coding: utf-8 -*-
import logging
import pandas as pd
from pathlib import Path

class FusionPipeline:
    def __init__(self, modality_dict):
        """
        modality_dict: dict of {name: (preprocessor, embedder)}
        """
        self.modality_dict = modality_dict
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def fit(self, X_dict):
        """X_dict is dict of DataFrames"""
        for name, (prep, emb) in self.modality_dict.items():
            if name in X_dict:
                print(f"DEBUG: Starting {name} preprocessor fit...")
                prep.fit(X_dict[name])
                print(f"DEBUG: Starting {name} preprocessor transform...")
                transformed_data = prep.transform(X_dict[name])
                print(f"DEBUG: Starting {name} embedder fit...")
                emb.fit(transformed_data)
                print(f"DEBUG: Finished {name} embedder fit.")
        return self
        
    def transform(self, X_dict):
        embeds = {}
        for name, (prep, emb) in self.modality_dict.items():
            if name in X_dict:
                proc = prep.transform(X_dict[name])
                embeds[name] = emb.transform(proc)
        return embeds
        
    def fuse(self, X_dict, output_dir: Path, labels_df=None, targets_df=None):
        embeds = self.transform(X_dict)
        
        # Inner join on PATNO
        patnos = None
        for name, df in embeds.items():
            if patnos is None:
                patnos = set(df.index)
            else:
                patnos = patnos.intersection(set(df.index))
                
        self.logger.info(f"Cohort size after joining: {len(patnos)}")
        
        fused = []
        for name in sorted(embeds.keys()):
            df = embeds[name].loc[list(patnos)]
            fused.append(df)
            
        fused_df = pd.concat(fused, axis=1)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        fused_df.to_parquet(output_dir / "fused_embeddings.parquet")
        
        if labels_df is not None:
            labels_df.loc[list(patnos)].to_parquet(output_dir / "labels.parquet")
            
        if targets_df is not None:
            targets_df.loc[list(patnos)].to_parquet(output_dir / "regression_targets.parquet")
            
        return fused_df
