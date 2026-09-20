import numpy as np
import torch

from src.models.mri_encoder import MRIEncoder


GRAPH_FEATURES = [
    "Brain_Stem",
    "Optic_Chiasm",

    "Left_Lateral_Ventricle",
    "Right_Lateral_Ventricle",
    "Left_Inf_Lat_Vent",
    "Right_Inf_Lat_Vent",
    "3rd_Ventricle",
    "4th_Ventricle",
    "5th_Ventricle",
    "Left_choroid_plexus",
    "Right_choroid_plexus",

    "Left_Cerebellum_Cortex",
    "Right_Cerebellum_Cortex",
    "Left_Cerebellum_White_Matter",
    "Right_Cerebellum_White_Matter",

    "Left_Amygdala",
    "Right_Amygdala",
    "Left_Accumbens_area",
    "Right_Accumbens_area",
    "Left_Caudate",
    "Right_Caudate",
    "Left_Hippocampus",
    "Right_Hippocampus",
    "Left_Pallidum",
    "Right_Pallidum",
    "Left_Putamen",
    "Right_Putamen",
    "Left_Thalamus",
    "Right_Thalamus",
    "Left_VentralDC",
    "Right_VentralDC",

    "CC_Anterior",
    "CC_Mid_Anterior",
    "CC_Central",
    "CC_Mid_Posterior",
    "CC_Posterior",
]


def test_mri_encoder_forward():

    feature_names = GRAPH_FEATURES + [
        "TotalGrayVol",
        "CortexVol",
        "BrainSegVol",
        "EstimatedTotalIntraCranialVol",
    ]

    encoder = MRIEncoder(
        input_dim=len(feature_names),
        embed_dim=32,
        dropout=0.2,
        feature_names=feature_names,
        graph_features=GRAPH_FEATURES,
        hidden_dim=64,
    )

    x = torch.randn(
        8,
        len(feature_names),
    )

    shared, private = encoder(x)

    assert shared.shape == (8, 32)
    assert private.shape == (8, 32)

    assert torch.isfinite(shared).all()
    assert torch.isfinite(private).all()


def test_mri_encoder_batch_size_one():

    feature_names = GRAPH_FEATURES + [
        "TotalGrayVol",
        "CortexVol",
        "BrainSegVol",
    ]

    encoder = MRIEncoder(
        input_dim=len(feature_names),
        embed_dim=32,
        dropout=0.2,
        feature_names=feature_names,
        graph_features=GRAPH_FEATURES,
        hidden_dim=64,
    )

    x = torch.randn(
        1,
        len(feature_names),
    )

    shared, private = encoder(x)

    assert shared.shape == (1, 32)
    assert private.shape == (1, 32)


def test_mri_encoder_deterministic_shape():

    feature_names = GRAPH_FEATURES + [
        "TotalGrayVol",
        "CortexVol",
    ]

    encoder = MRIEncoder(
        input_dim=len(feature_names),
        embed_dim=16,
        dropout=0.0,
        feature_names=feature_names,
        graph_features=GRAPH_FEATURES,
        hidden_dim=32,
    )

    encoder.eval()

    x = torch.randn(
        4,
        len(feature_names),
    )

    shared, private = encoder(x)

    assert shared.shape == (4, 16)
    assert private.shape == (4, 16)


# # -*- coding: utf-8 -*-
# import pytest
# import numpy as np
# import pandas as pd

# from src.embeddings.clinical_embedder import ClinicalEmbedder
# from src.embeddings.mri_embedder import MRIEmbedder
# from src.embeddings.pet_embedder import PETEmbedder
# from src.embeddings.genetic_embedder import GeneticEmbedder
# from src.embeddings.fusion import FusionPipeline

# class TestEmbedders:
#     def test_clinical_pca(self):
#         emb = ClinicalEmbedder(mode='pca')
#         data = pd.DataFrame(np.random.randn(50, 20))
#         out = emb.fit_transform(data)
#         assert out.shape == (50, 32)
        
#     def test_mri_pca(self):
#         emb = MRIEmbedder(mode='pca')
#         data = pd.DataFrame(np.random.randn(50, 100))
#         out = emb.fit_transform(data)
#         assert out.shape == (50, 64)
        
#     def test_pet_padding(self):
#         emb = PETEmbedder()
#         data = pd.DataFrame(np.random.randn(50, 10))
#         out = emb.fit_transform(data)
#         assert out.shape == (50, 16)
        
#     def test_genetic_padding(self):
#         emb = GeneticEmbedder()
#         data = pd.DataFrame(np.random.randn(50, 9))
#         out = emb.fit_transform(data)
#         assert out.shape == (50, 32)

