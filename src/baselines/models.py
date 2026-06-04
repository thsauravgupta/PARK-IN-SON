# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from lightgbm import LGBMRegressor, LGBMClassifier

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

class ModelFactory:
    @staticmethod
    def get_model(name: str, task: str, config: dict):
        seed = config.get("seed", 42)
        
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
            
        else:
            raise ValueError(f"Unknown model {name}")
