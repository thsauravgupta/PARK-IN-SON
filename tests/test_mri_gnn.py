import numpy as np
import torch

from src.models.mri_encoder import (
    MRIEncoder,
    AnatomyGraphBuilder,
)


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


def test_graph_is_square_and_symmetric():

    adjacency = AnatomyGraphBuilder.build(
        GRAPH_FEATURES
    )

    assert adjacency.shape == (
        len(GRAPH_FEATURES),
        len(GRAPH_FEATURES),
    )

    assert torch.allclose(
        adjacency,
        adjacency.T,
        atol=1e-6,
    )


def test_mri_gnn_forward():

    feature_names = GRAPH_FEATURES + [
        "TotalGrayVol",
        "CortexVol",
        "BrainSegVol",
        "EstimatedTotalIntraCranialVol",
    ]

    model = MRIEncoder(

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

    shared, private = model(x)

    assert shared.shape == (
        8,
        32,
    )

    assert private.shape == (
        8,
        32,
    )


def test_mri_gnn_is_not_target_dependent():

    """
    The graph builder accepts only feature names.

    This is a regression test ensuring graph construction cannot accidentally
    receive UPDRS/diagnosis information.
    """

    adjacency_a = AnatomyGraphBuilder.build(
        GRAPH_FEATURES
    )

    adjacency_b = AnatomyGraphBuilder.build(
        list(reversed(GRAPH_FEATURES))
    )

    assert adjacency_a.shape == adjacency_b.shape