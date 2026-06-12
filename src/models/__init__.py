# -*- coding: utf-8 -*-
"""
src.models — Deep learning model components for PARK-IN-SON.

Public API
----------
MultimodalEncoder   Top-level orchestrator (encoders + fusion + GNN)
FusionTransformer   Cross-modal attention fusion
PatientGNN          GATv2 patient similarity graph
BaseEncoder         Abstract base for all modality encoders

Modality encoders (in src.models.encoders):
    ClinicalEncoder     MLP autoencoder for tabular clinical features
    ROITransformerEncoder  Transformer over brain ROI tokens (MRI)
    PETEncoder          Lightweight MLP for DaTscan features
    GeneticEncoder      Gene token embedding + Transformer
"""

from src.models.multimodal_encoder import MultimodalEncoder
from src.models.fusion.fusion_transformer import FusionTransformer
from src.models.fusion.patient_gnn import PatientGNN
from src.models.encoders.base_encoder import BaseEncoder
from src.models.encoders.clinical_encoder import ClinicalEncoder
from src.models.encoders.mri_encoder import ROITransformerEncoder
from src.models.encoders.pet_encoder import PETEncoder
from src.models.encoders.genetic_encoder import GeneticEncoder

__all__ = [
    "MultimodalEncoder",
    "FusionTransformer",
    "PatientGNN",
    "BaseEncoder",
    "ClinicalEncoder",
    "ROITransformerEncoder",
    "PETEncoder",
    "GeneticEncoder",
]
