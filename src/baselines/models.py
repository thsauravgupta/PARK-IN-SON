# -*- coding: utf-8 -*-
"""
Baseline models for Parkinson's disease prediction.

Includes:
  - Traditional ML: random_forest, xgboost, svm, ridge, lightgbm,
                     knn, adaboost, naive_bayes, elastic_net
  - Deep Learning:  mlp, cnn_1d, tabnet
  - Graph-based:    gnn_embedding, federated_gnn
"""
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
)
from xgboost import XGBRegressor, XGBClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import (
    RidgeCV, LogisticRegressionCV,
    ElasticNet, SGDClassifier,
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMRegressor, LGBMClassifier


# =============================================================================
# MLP (original)
# =============================================================================

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim=144, output_dim=1, is_classification=False):
        super().__init__()
        self.is_classification = is_classification
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        out = self.net(x)
        if self.is_classification:
            return torch.sigmoid(out)
        return out

class CustomMLPEstimator(BaseEstimator):
    def __init__(self, input_dim=144, is_classification=False, epochs=200, lr=1e-3, batch_size=64, patience=10):
        self.input_dim = input_dim
        self.is_classification = is_classification
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        # Force CPU to avoid PyTorch deadlock on Windows
        self.device = torch.device("cpu")
        self.model = None

    def fit(self, X, y):
        # Holdout 10% for validation
        n_val = max(int(0.1 * len(X)), 1)
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).view(-1, 1))
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        self.model = PyTorchMLP(input_dim=self.input_dim, is_classification=self.is_classification).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss() if self.is_classification else nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            self.model.train()
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    out = self.model(bx)
                    val_loss += criterion(out, by).item()
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                break
                
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X).to(self.device)).cpu().numpy().flatten()
        if self.is_classification:
            return (out > 0.5).astype(int)
        return out

    def predict_proba(self, X):
        if not self.is_classification:
            raise NotImplementedError
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X).to(self.device)).cpu().numpy().flatten()
        return np.vstack([1-out, out]).T

class MLPRegressor(CustomMLPEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        super().__init__(is_classification=False, **kwargs)

class MLPClassifier(CustomMLPEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        super().__init__(is_classification=True, **kwargs)


# =============================================================================
# CNN-1D: treats 144-dim embedding as a 1D signal
# =============================================================================

class CNN1DNet(nn.Module):
    """1D-CNN that reshapes (batch, 144) -> (batch, 1, 144) and applies convolutions."""

    def __init__(self, input_dim=144, output_dim=1, is_classification=False):
        super().__init__()
        self.input_dim = input_dim
        self.is_classification = is_classification

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),    # -> (batch, 32, 144)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),                   # -> (batch, 32, 72)
            nn.Conv1d(32, 64, kernel_size=3, padding=1),   # -> (batch, 64, 72)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                       # -> (batch, 64, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        # x: (batch, input_dim) -> (batch, 1, input_dim)
        x = x.unsqueeze(1)
        x = self.features(x)           # (batch, 64, 1)
        x = x.squeeze(-1)              # (batch, 64)
        out = self.head(x)
        if self.is_classification:
            return torch.sigmoid(out)
        return out


class CNN1DEstimator(BaseEstimator):
    """Sklearn-compatible wrapper for CNN1DNet."""

    def __init__(self, input_dim=144, is_classification=False,
                 epochs=200, lr=1e-3, batch_size=64, patience=15):
        self.input_dim = input_dim
        self.is_classification = is_classification
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.device = torch.device("cpu")
        self.model = None

    def fit(self, X, y):
        n_val = max(int(0.1 * len(X)), 1)
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).view(-1, 1))

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        self.model = CNN1DNet(
            input_dim=self.input_dim,
            is_classification=self.is_classification,
        ).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss() if self.is_classification else nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for _ in range(self.epochs):
            self.model.train()
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(bx), by)
                loss.backward()
                optimizer.step()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    val_loss += criterion(self.model(bx), by).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X).to(self.device)).cpu().numpy().flatten()
        if self.is_classification:
            return (out > 0.5).astype(int)
        return out

    def predict_proba(self, X):
        if not self.is_classification:
            raise NotImplementedError
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X).to(self.device)).cpu().numpy().flatten()
        return np.vstack([1 - out, out]).T


class CNN1DRegressor(CNN1DEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        super().__init__(is_classification=False, **kwargs)


class CNN1DClassifier(CNN1DEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        super().__init__(is_classification=True, **kwargs)


# =============================================================================
# TabNet (custom lightweight implementation)
# =============================================================================

class TabNetModule(nn.Module):
    """
    Simplified TabNet with 2 decision steps.
    Uses softmax + top-k masking as an approximate Sparsemax.
    """

    def __init__(self, input_dim=144, output_dim=1, n_steps=2,
                 hidden_dim=128, step_output_dim=64, top_k=10,
                 is_classification=False):
        super().__init__()
        self.n_steps = n_steps
        self.top_k = top_k
        self.is_classification = is_classification

        # Shared initial batch-norm
        self.initial_bn = nn.BatchNorm1d(input_dim)

        # Per-step networks
        self.step_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        for _ in range(n_steps):
            self.step_layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, step_output_dim),
            ))
            self.attention_layers.append(nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.BatchNorm1d(input_dim),
            ))

        self.final = nn.Linear(step_output_dim, output_dim)

    def _sparsemax_approx(self, logits):
        """Softmax followed by top-k masking to approximate sparsemax."""
        probs = torch.softmax(logits, dim=-1)
        if self.top_k < logits.size(-1):
            topk_vals, topk_idx = probs.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(probs)
            mask.scatter_(-1, topk_idx, 1.0)
            probs = probs * mask
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-15)
        return probs

    def forward(self, x):
        x = self.initial_bn(x)
        aggregated = torch.zeros(x.size(0), self.step_layers[0][-1].out_features,
                                 device=x.device)

        for step in range(self.n_steps):
            # Attention mask
            attn_logits = self.attention_layers[step](x)
            attn = self._sparsemax_approx(attn_logits)
            masked_x = attn * x
            # Step transform
            step_out = self.step_layers[step](masked_x)
            aggregated = aggregated + torch.relu(step_out)

        out = self.final(aggregated)
        if self.is_classification:
            return torch.sigmoid(out)
        return out


class TabNetEstimator(BaseEstimator):
    """Sklearn-compatible wrapper for TabNetModule."""

    def __init__(self, input_dim=144, is_classification=False,
                 epochs=200, lr=1e-3, batch_size=64, patience=15):
        self.input_dim = input_dim
        self.is_classification = is_classification
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.device = torch.device("cpu")
        self.model = None

    def fit(self, X, y):
        n_val = max(int(0.1 * len(X)), 1)
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).view(-1, 1))

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        self.model = TabNetModule(
            input_dim=self.input_dim,
            is_classification=self.is_classification,
        ).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss() if self.is_classification else nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for _ in range(self.epochs):
            self.model.train()
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(bx), by)
                loss.backward()
                optimizer.step()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    val_loss += criterion(self.model(bx), by).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X).to(self.device)).cpu().numpy().flatten()
        if self.is_classification:
            return (out > 0.5).astype(int)
        return out

    def predict_proba(self, X):
        if not self.is_classification:
            raise NotImplementedError
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(X).to(self.device)).cpu().numpy().flatten()
        return np.vstack([1 - out, out]).T


class TabNetRegressor(TabNetEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        super().__init__(is_classification=False, **kwargs)


class TabNetClassifier(TabNetEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        super().__init__(is_classification=True, **kwargs)


# =============================================================================
# GNN Embedding: builds a patient-similarity KNN graph, then runs a 2-layer GCN
# =============================================================================

class _GCNNet(nn.Module):
    """2-layer GCN for node-level prediction. Requires torch_geometric."""

    def __init__(self, input_dim=144, hidden_dim=128, embed_dim=64,
                 output_dim=1, is_classification=False):
        super().__init__()
        from torch_geometric.nn import GCNConv  # lazy import

        self.is_classification = is_classification
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        out = self.head(x)
        if self.is_classification:
            return torch.sigmoid(out)
        return out


def _build_knn_graph(X, k=5):
    """Build a symmetric KNN graph and return edge_index (2, num_edges) as LongTensor."""
    from sklearn.neighbors import NearestNeighbors

    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
    nn_model.fit(X)
    distances, indices = nn_model.kneighbors(X)

    src, dst = [], []
    for i in range(len(X)):
        for j in indices[i, 1:]:        # skip self
            src.extend([i, int(j)])
            dst.extend([int(j), i])
    edge_index = torch.LongTensor([src, dst])
    # Remove duplicate edges
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


class GNNEmbeddingEstimator(BaseEstimator):
    """Sklearn-compatible GCN that builds a patient-similarity KNN graph."""

    def __init__(self, input_dim=144, k=5, is_classification=False,
                 epochs=200, lr=1e-3, patience=15):
        self.input_dim = input_dim
        self.k = k
        self.is_classification = is_classification
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.device = torch.device("cpu")
        self.model = None
        self._X_train = None  # stored for predict-time graph building

    def fit(self, X, y):
        self._X_train = np.array(X, dtype=np.float32)
        edge_index = _build_knn_graph(self._X_train, k=self.k)

        x_tensor = torch.FloatTensor(self._X_train).to(self.device)
        y_tensor = torch.FloatTensor(y).view(-1, 1).to(self.device)
        edge_index = edge_index.to(self.device)

        self.model = _GCNNet(
            input_dim=self.input_dim,
            is_classification=self.is_classification,
        ).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss() if self.is_classification else nn.MSELoss()

        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        for _ in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(x_tensor, edge_index)
            loss = criterion(out, y_tensor)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def _predict_raw(self, X):
        """Run inference on X by building a joint graph with stored training data."""
        from sklearn.neighbors import NearestNeighbors

        X = np.array(X, dtype=np.float32)
        n_train = len(self._X_train)
        n_test = len(X)
        X_all = np.vstack([self._X_train, X])

        # Build KNN graph over all nodes (train + test)
        edge_index = _build_knn_graph(X_all, k=self.k).to(self.device)
        x_tensor = torch.FloatTensor(X_all).to(self.device)

        self.model.eval()
        with torch.no_grad():
            out = self.model(x_tensor, edge_index).cpu().numpy().flatten()
        # Return only the test-node predictions
        return out[n_train:]

    def predict(self, X):
        out = self._predict_raw(X)
        if self.is_classification:
            return (out > 0.5).astype(int)
        return out

    def predict_proba(self, X):
        if not self.is_classification:
            raise NotImplementedError
        out = self._predict_raw(X)
        return np.vstack([1 - out, out]).T


class GNNRegressor(GNNEmbeddingEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        super().__init__(is_classification=False, **kwargs)


class GNNClassifier(GNNEmbeddingEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        super().__init__(is_classification=True, **kwargs)


# =============================================================================
# Federated GNN: simulated FedAvg with 3 GCN clients
# =============================================================================

class FederatedGNNEstimator(BaseEstimator):
    """
    Simulated Federated Learning with 3 GNN clients using FedAvg.

    Each client owns a random partition of the training data, trains its own
    GCN locally for `local_epochs`, then all client weights are averaged.
    This is repeated for `global_rounds`.
    """

    def __init__(self, input_dim=144, k=5, is_classification=False,
                 global_rounds=20, local_epochs=5, lr=1e-3, n_clients=3):
        self.input_dim = input_dim
        self.k = k
        self.is_classification = is_classification
        self.global_rounds = global_rounds
        self.local_epochs = local_epochs
        self.lr = lr
        self.n_clients = n_clients
        self.device = torch.device("cpu")
        self.model = None
        self._X_train = None

    def fit(self, X, y):
        self._X_train = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # Randomly partition data into clients
        indices = np.random.permutation(len(X))
        client_splits = np.array_split(indices, self.n_clients)

        # Initialise one model per client (same architecture)
        client_models = []
        for _ in range(self.n_clients):
            m = _GCNNet(
                input_dim=self.input_dim,
                is_classification=self.is_classification,
            ).to(self.device)
            client_models.append(m)

        criterion = nn.BCELoss() if self.is_classification else nn.MSELoss()

        # Global model (used to seed each round)
        global_model = _GCNNet(
            input_dim=self.input_dim,
            is_classification=self.is_classification,
        ).to(self.device)

        for _round in range(self.global_rounds):
            # Broadcast global weights to all clients
            global_state = global_model.state_dict()
            for cm in client_models:
                cm.load_state_dict(copy.deepcopy(global_state))

            # Local training
            for ci, cm in enumerate(client_models):
                idx = client_splits[ci]
                if len(idx) < 2:
                    continue
                X_c = self._X_train[idx]
                y_c = y[idx]

                edge_index = _build_knn_graph(X_c, k=min(self.k, len(X_c) - 1)).to(self.device)
                x_t = torch.FloatTensor(X_c).to(self.device)
                y_t = torch.FloatTensor(y_c).view(-1, 1).to(self.device)

                opt = optim.Adam(cm.parameters(), lr=self.lr)
                cm.train()
                for _ in range(self.local_epochs):
                    opt.zero_grad()
                    loss = criterion(cm(x_t, edge_index), y_t)
                    loss.backward()
                    opt.step()

            # FedAvg: average all client weights
            avg_state = {}
            for key in global_state:
                avg_state[key] = torch.stack(
                    [cm.state_dict()[key].float() for cm in client_models]
                ).mean(dim=0)
            global_model.load_state_dict(avg_state)

        self.model = global_model
        return self

    def _predict_raw(self, X):
        X = np.array(X, dtype=np.float32)
        n_train = len(self._X_train)
        X_all = np.vstack([self._X_train, X])

        edge_index = _build_knn_graph(X_all, k=self.k).to(self.device)
        x_tensor = torch.FloatTensor(X_all).to(self.device)

        self.model.eval()
        with torch.no_grad():
            out = self.model(x_tensor, edge_index).cpu().numpy().flatten()
        return out[n_train:]

    def predict(self, X):
        out = self._predict_raw(X)
        if self.is_classification:
            return (out > 0.5).astype(int)
        return out

    def predict_proba(self, X):
        if not self.is_classification:
            raise NotImplementedError
        out = self._predict_raw(X)
        return np.vstack([1 - out, out]).T


class FederatedGNNRegressor(FederatedGNNEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        super().__init__(is_classification=False, **kwargs)


class FederatedGNNClassifier(FederatedGNNEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        super().__init__(is_classification=True, **kwargs)


# =============================================================================
# Model Factory
# =============================================================================

class ModelFactory:
    """Central registry that returns (model, param_grid) for any supported name."""

    @staticmethod
    def get_model(name: str, task: str, config: dict):
        seed = config.get("seed", 42)

        # --- Original models (unchanged) ---

        if name == 'random_forest':
            if task == 'regression':
                model = RandomForestRegressor(n_estimators=200, random_state=seed)
            else:
                model = RandomForestClassifier(n_estimators=200, random_state=seed)
            param_grid = {
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            }
            return model, param_grid

        elif name == 'xgboost':
            if task == 'regression':
                model = XGBRegressor(n_estimators=300, early_stopping_rounds=20, random_state=seed)
            else:
                model = XGBClassifier(n_estimators=300, early_stopping_rounds=20, random_state=seed, use_label_encoder=False)
            param_grid = {
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6]
            }
            return model, param_grid

        elif name == 'svm':
            if task == 'regression':
                model = SVR()
            else:
                model = SVC(probability=True, random_state=seed)
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto']
            }
            return model, param_grid

        elif name == 'ridge':
            if task == 'regression':
                model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
                return model, {}
            else:
                model = LogisticRegressionCV(random_state=seed)
                return model, {}

        elif name == 'mlp':
            if task == 'regression':
                model = MLPRegressor()
            else:
                model = MLPClassifier()
            return model, {}

        elif name == 'lightgbm':
            if task == 'regression':
                model = LGBMRegressor(random_state=seed)
            else:
                model = LGBMClassifier(random_state=seed)
            param_grid = {
                'num_leaves': [31, 63],
                'learning_rate': [0.01, 0.1],
                'n_estimators': [100, 300]
            }
            return model, param_grid

        # --- New models ---

        elif name == 'knn':
            if task == 'regression':
                model = KNeighborsRegressor()
            else:
                model = KNeighborsClassifier()
            param_grid = {
                'n_neighbors': [3, 5, 7, 11],
                'weights': ['uniform', 'distance'],
            }
            return model, param_grid

        elif name == 'adaboost':
            if task == 'regression':
                model = AdaBoostRegressor(random_state=seed)
            else:
                model = AdaBoostClassifier(random_state=seed)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0],
            }
            return model, param_grid

        elif name == 'naive_bayes':
            if task == 'regression':
                # GaussianNB is classification-only; fall back to KNN for regression
                model = KNeighborsRegressor()
                return model, {}
            else:
                model = GaussianNB()
                return model, {}

        elif name == 'elastic_net':
            if task == 'regression':
                model = ElasticNet(random_state=seed)
            else:
                # SGDClassifier with log_loss behaves like a regularised logistic regression
                model = SGDClassifier(loss='log_loss', random_state=seed)
            param_grid = {
                'alpha': [0.01, 0.1, 1.0],
                'l1_ratio': [0.2, 0.5, 0.8],
            }
            return model, param_grid

        elif name == 'cnn_1d':
            if task == 'regression':
                model = CNN1DRegressor()
            else:
                model = CNN1DClassifier()
            return model, {}

        elif name == 'gnn_embedding':
            if task == 'regression':
                model = GNNRegressor()
            else:
                model = GNNClassifier()
            return model, {}

        elif name == 'federated_gnn':
            if task == 'regression':
                model = FederatedGNNRegressor()
            else:
                model = FederatedGNNClassifier()
            return model, {}

        elif name == 'tabnet':
            if task == 'regression':
                model = TabNetRegressor()
            else:
                model = TabNetClassifier()
            return model, {}

        else:
            raise ValueError(f"Unknown model {name}")
