# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pandas as pd

from src.embeddings.clinical_embedder import ClinicalEmbedder
from src.embeddings.mri_embedder import MRIEmbedder
from src.embeddings.pet_embedder import PETEmbedder
from src.embeddings.genetic_embedder import GeneticEmbedder
from src.embeddings.fusion import FusionPipeline

class TestEmbedders:
    def test_clinical_pca(self):
        emb = ClinicalEmbedder(mode='pca')
        data = pd.DataFrame(np.random.randn(50, 20))
        out = emb.fit_transform(data)
        assert out.shape == (50, 32)
        
    def test_mri_pca(self):
        emb = MRIEmbedder(mode='pca')
        data = pd.DataFrame(np.random.randn(50, 100))
        out = emb.fit_transform(data)
        assert out.shape == (50, 64)
        
    def test_pet_padding(self):
        emb = PETEmbedder()
        data = pd.DataFrame(np.random.randn(50, 10))
        out = emb.fit_transform(data)
        assert out.shape == (50, 16)
        
    def test_genetic_padding(self):
        emb = GeneticEmbedder()
        data = pd.DataFrame(np.random.randn(50, 9))
        out = emb.fit_transform(data)
        assert out.shape == (50, 32)
