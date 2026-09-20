# -*- coding: utf-8 -*-
"""
Anatomy-informed GNN encoder for PPMI FreeSurfer MRI features.

The PPMI FreeSurfer table contains:
    1. Regional structural measurements.
    2. Global MRI summary measurements.

Regional measurements are represented as nodes in a fixed,
label-independent anatomy-informed graph.

Global measurements are encoded separately with an MLP.

The two representations are combined into the MRI shared/private
latent representations used by Fed-PhenoGraft.

Important:
    The graph topology is fixed before training and does not use
    UPDRS scores, diagnosis labels, validation data, or test data.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedGraphConv(nn.Module):
    """
    Basic graph convolution:

        H' = A_norm H W

    A_norm is fixed and constructed once from the anatomical graph.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:

        # x:
        #   [batch, number_of_nodes, feature_dimension]

        x = torch.matmul(adjacency, x)

        return self.linear(x)


class AnatomyGraphBuilder:
    """
    Creates the fixed anatomy-informed graph.

    The topology is based on:
      - bilateral homologous structures
      - anatomical families
      - sparse known neuroanatomical relationships

    No patient measurements or prediction targets are used to build it.
    """

    @staticmethod
    def build(feature_names: Sequence[str]) -> torch.Tensor:

        names = list(feature_names)
        n = len(names)

        # Self loops.
        adjacency = torch.eye(
            n,
            dtype=torch.float32,
        )

        index = {
            name: i
            for i, name in enumerate(names)
        }

        # ---------------------------------------------------------------
        # Bilateral homologous connections
        # ---------------------------------------------------------------

        bilateral_pairs = [
            ("Left_Amygdala", "Right_Amygdala"),
            ("Left_Accumbens_area", "Right_Accumbens_area"),
            ("Left_Caudate", "Right_Caudate"),
            ("Left_Hippocampus", "Right_Hippocampus"),
            ("Left_Pallidum", "Right_Pallidum"),
            ("Left_Putamen", "Right_Putamen"),
            ("Left_Thalamus", "Right_Thalamus"),
            ("Left_VentralDC", "Right_VentralDC"),

            ("Left_Lateral_Ventricle", "Right_Lateral_Ventricle"),
            ("Left_Inf_Lat_Vent", "Right_Inf_Lat_Vent"),

            ("Left_choroid_plexus", "Right_choroid_plexus"),

            ("Left_Cerebellum_Cortex", "Right_Cerebellum_Cortex"),
            (
                "Left_Cerebellum_White_Matter",
                "Right_Cerebellum_White_Matter",
            ),
        ]

        for left, right in bilateral_pairs:

            if left in index and right in index:

                i = index[left]
                j = index[right]

                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0

        # ---------------------------------------------------------------
        # Anatomical families
        # ---------------------------------------------------------------

        families = {

            "subcortical": [
                name
                for name in names
                if any(
                    token in name
                    for token in [
                        "Amygdala",
                        "Accumbens",
                        "Caudate",
                        "Hippocampus",
                        "Pallidum",
                        "Putamen",
                        "Thalamus",
                        "VentralDC",
                    ]
                )
            ],

            "ventricular": [
                name
                for name in names
                if (
                    "Ventricle" in name
                    or "choroid_plexus" in name
                )
            ],

            "cerebellar": [
                name
                for name in names
                if "Cerebellum" in name
            ],

            "midline": [
                name
                for name in names
                if (
                    name in {
                        "Brain_Stem",
                        "Optic_Chiasm",
                    }
                    or name.startswith("CC_")
                )
            ],
        }

        # Connect structures belonging to the same anatomical family.
        for members in families.values():

            for i_pos in range(len(members)):

                for j_pos in range(i_pos + 1, len(members)):

                    i = index[members[i_pos]]
                    j = index[members[j_pos]]

                    adjacency[i, j] = max(
                        float(adjacency[i, j]),
                        0.35,
                    )

                    adjacency[j, i] = max(
                        float(adjacency[j, i]),
                        0.35,
                    )

        # ---------------------------------------------------------------
        # Sparse cross-system anatomical connections
        # ---------------------------------------------------------------

        cross_links = [

            ("Brain_Stem", "Left_Thalamus"),
            ("Brain_Stem", "Right_Thalamus"),

            ("Brain_Stem", "Left_VentralDC"),
            ("Brain_Stem", "Right_VentralDC"),

            ("Left_Thalamus", "Left_Caudate"),
            ("Right_Thalamus", "Right_Caudate"),

            ("Left_Caudate", "Left_Putamen"),
            ("Right_Caudate", "Right_Putamen"),

            ("Left_Putamen", "Left_Pallidum"),
            ("Right_Putamen", "Right_Pallidum"),

            ("Left_Hippocampus", "Left_Amygdala"),
            ("Right_Hippocampus", "Right_Amygdala"),

            ("Left_Cerebellum_Cortex", "Left_Thalamus"),
            ("Right_Cerebellum_Cortex", "Right_Thalamus"),

            ("3rd_Ventricle", "Left_Thalamus"),
            ("3rd_Ventricle", "Right_Thalamus"),
        ]

        for first, second in cross_links:

            if first in index and second in index:

                i = index[first]
                j = index[second]

                adjacency[i, j] = max(
                    float(adjacency[i, j]),
                    0.50,
                )

                adjacency[j, i] = max(
                    float(adjacency[j, i]),
                    0.50,
                )

        # ---------------------------------------------------------------
        # Symmetric GCN normalization
        #
        # A_norm = D^(-1/2) A D^(-1/2)
        # ---------------------------------------------------------------

        degree = adjacency.sum(dim=1).clamp_min(1e-8)

        inverse_sqrt_degree = degree.pow(-0.5)

        normalized = (
            inverse_sqrt_degree[:, None]
            * adjacency
            * inverse_sqrt_degree[None, :]
        )

        return normalized


class MRIAnatomicalGNN(nn.Module):
    """
    Anatomy-informed GNN for regional FreeSurfer measurements.

    Each node initially contains:
        [regional_volume, learnable_node_identity]

    This allows the network to distinguish, for example,
    hippocampus from putamen even though both begin as scalar volumes.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        dropout: float,
        feature_names: Sequence[str],
        graph_features: Sequence[str],
        hidden_dim: int = 64,
        node_id_dim: int = 8,
    ):

        super().__init__()

        self.input_dim = int(input_dim)

        self.feature_names = list(feature_names)
        self.graph_features = list(graph_features)

        missing = [
            feature
            for feature in self.graph_features
            if feature not in self.feature_names
        ]

        if missing:

            raise ValueError(
                "The following MRI graph features are missing from the "
                f"FreeSurfer table: {missing}"
            )

        # Position of graph features in the complete MRI vector.
        self.graph_indices = [
            self.feature_names.index(feature)
            for feature in self.graph_features
        ]

        graph_feature_set = set(self.graph_features)

        # Everything else is treated as global/contextual MRI information.
        self.global_indices = [
            i
            for i, feature in enumerate(self.feature_names)
            if feature not in graph_feature_set
        ]

        adjacency = AnatomyGraphBuilder.build(
            self.graph_features
        )

        self.register_buffer(
            "adjacency",
            adjacency,
        )

        # ---------------------------------------------------------------
        # Learnable node identities
        # ---------------------------------------------------------------

        self.node_identity = nn.Parameter(
            torch.randn(
                len(self.graph_features),
                node_id_dim,
            ) * 0.02
        )

        # ---------------------------------------------------------------
        # GCN
        # ---------------------------------------------------------------

        self.gcn1 = FixedGraphConv(
            1 + node_id_dim,
            hidden_dim,
        )

        self.gcn2 = FixedGraphConv(
            hidden_dim,
            hidden_dim,
        )

        self.gcn_norm = nn.LayerNorm(
            hidden_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        # ---------------------------------------------------------------
        # Global MRI branch
        # ---------------------------------------------------------------

        global_dim = max(
            1,
            len(self.global_indices),
        )

        self.global_encoder = nn.Sequential(

            nn.Linear(
                global_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.GELU(),
        )

        # Graph mean + graph max + global representation.
        combined_dim = hidden_dim * 3

        # ---------------------------------------------------------------
        # Shared MRI representation
        # ---------------------------------------------------------------

        self.shared_head = nn.Sequential(

            nn.Linear(
                combined_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                embed_dim,
            ),
        )

        # ---------------------------------------------------------------
        # Private MRI representation
        # ---------------------------------------------------------------

        self.private_head = nn.Sequential(

            nn.Linear(
                combined_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                embed_dim,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ):

        # ---------------------------------------------------------------
        # Extract regional MRI values
        # ---------------------------------------------------------------

        regional = x[
            :,
            self.graph_indices
        ].unsqueeze(-1)

        # [batch, nodes, node_identity_dim]
        node_identity = self.node_identity.unsqueeze(0).expand(
            x.size(0),
            -1,
            -1,
        )

        node_input = torch.cat(
            [
                regional,
                node_identity,
            ],
            dim=-1,
        )

        # ---------------------------------------------------------------
        # GCN layer 1
        # ---------------------------------------------------------------

        h = self.gcn1(
            node_input,
            self.adjacency,
        )

        h = F.gelu(h)
        h = self.dropout(h)

        # ---------------------------------------------------------------
        # GCN layer 2
        # ---------------------------------------------------------------

        h = self.gcn2(
            h,
            self.adjacency,
        )

        h = self.gcn_norm(h)
        h = F.gelu(h)
        h = self.dropout(h)

        # ---------------------------------------------------------------
        # Graph-level pooling
        # ---------------------------------------------------------------

        graph_mean = h.mean(
            dim=1
        )

        graph_max = h.max(
            dim=1
        ).values

        # ---------------------------------------------------------------
        # Global MRI information
        # ---------------------------------------------------------------

        if self.global_indices:

            global_x = x[
                :,
                self.global_indices
            ]

        else:

            global_x = x.new_zeros(
                (
                    x.size(0),
                    1,
                )
            )

        global_embedding = self.global_encoder(
            global_x
        )

        # ---------------------------------------------------------------
        # Combine regional + global MRI information
        # ---------------------------------------------------------------

        combined = torch.cat(
            [
                graph_mean,
                graph_max,
                global_embedding,
            ],
            dim=-1,
        )

        shared = self.shared_head(
            combined
        )

        private = self.private_head(
            combined
        )

        return shared, private


class MRIEncoder(nn.Module):
    """
    Wrapper around the MRI encoder.

    When graph feature names are provided:
        FreeSurfer → anatomy-informed GNN.

    Otherwise:
        fallback MLP, useful for NIfTI/legacy representations.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        dropout: float,
        feature_names=None,
        graph_features=None,
        hidden_dim: int = 64,
    ):

        super().__init__()

        self.uses_graph = bool(
            feature_names
            and graph_features
        )

        if self.uses_graph:

            self.encoder = MRIAnatomicalGNN(

                input_dim=input_dim,

                embed_dim=embed_dim,

                dropout=dropout,

                feature_names=feature_names,

                graph_features=graph_features,

                hidden_dim=hidden_dim,
            )

        else:

            self.encoder = nn.ModuleDict({

                "shared": nn.Sequential(

                    nn.Linear(
                        input_dim,
                        hidden_dim,
                    ),

                    nn.GELU(),

                    nn.Dropout(
                        dropout
                    ),

                    nn.Linear(
                        hidden_dim,
                        embed_dim,
                    ),
                ),

                "private": nn.Sequential(

                    nn.Linear(
                        input_dim,
                        hidden_dim,
                    ),

                    nn.GELU(),

                    nn.Dropout(
                        dropout
                    ),

                    nn.Linear(
                        hidden_dim,
                        embed_dim,
                    ),
                ),
            })

    def forward(
        self,
        x: torch.Tensor,
    ):

        if self.uses_graph:

            return self.encoder(x)

        return (
            self.encoder["shared"](x),
            self.encoder["private"](x),
        )