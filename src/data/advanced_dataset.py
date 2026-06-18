# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

class PPMIFusedDataset(Dataset):
    """
    A unified PyTorch Dataset to load the fused graph embeddings.
    Designed to be plugged directly into advanced architectures:
    - Dual Twin Networks
    - Federated Learning Clients
    - Cross-Attention Transformers
    """
    def __init__(self, data_dir: str, task: str = 'classification', is_train: bool = True, scaler=None):
        """
        Args:
            data_dir: Path to the data/embeddings directory
            task: 'classification' (predicts diagnosis) or 'regression' (predicts UPDRS)
            is_train: Boolean indicating if this is training data (for scaling)
            scaler: Optional pre-fit StandardScaler (pass the training scaler to the test dataset)
        """
        self.data_dir = Path(data_dir)
        self.task = task
        
        # 1. Load the fused embeddings (The latent baseline data)
        X_df = pd.read_parquet(self.data_dir / "fused_embeddings.parquet")
        
        # 2. Load the targets
        if task == 'classification':
            y_df = pd.read_parquet(self.data_dir / "labels.parquet").squeeze()
        else:
            y_df = pd.read_parquet(self.data_dir / "regression_targets.parquet").squeeze()
            
        # Align indices to ensure no mismatch
        common_idx = X_df.index.intersection(y_df.index)
        X_df = X_df.loc[common_idx]
        self.labels = y_df.loc[common_idx].values
        
        # 3. Scaling (Critical for Deep Learning)
        if scaler is None:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(X_df.values) if is_train else X_df.values
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(X_df.values)
            
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.features[idx])
        y = torch.FloatTensor([self.labels[idx]])
        return x, y

def get_federated_dataloaders(data_dir: str, num_clients: int = 3, batch_size: int = 32):
    """
    Example utility to split the baseline dataset into multiple dataloaders
    for a Federated Learning simulation.
    """
    full_dataset = PPMIFusedDataset(data_dir, task='classification', is_train=True)
    
    # Calculate split sizes
    base_size = len(full_dataset) // num_clients
    sizes = [base_size] * num_clients
    sizes[-1] += len(full_dataset) - sum(sizes) # Add remainder to last client
    
    # Randomly split the dataset
    datasets = torch.utils.data.random_split(full_dataset, sizes)
    
    # Create a dataloader for each federated client
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in datasets]
    return client_loaders

# Example Usage:
# if __name__ == "__main__":
#     dataset = PPMIFusedDataset(data_dir="../../data/embeddings")
#     loader = DataLoader(dataset, batch_size=64, shuffle=True)
#     for batch_x, batch_y in loader:
#         print("Input shape:", batch_x.shape) # Should be [64, 144]
#         break
