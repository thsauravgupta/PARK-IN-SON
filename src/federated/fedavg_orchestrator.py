import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.evaluation.metrics import concordance_correlation_coefficient
import numpy as np

def fedavg_aggregate(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

def client_update(client_model, dataloader, epochs, device='cpu'):
    client_model.train()
    optimizer = optim.Adam(client_model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(epochs):
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            out = client_model(batch)
            
            loss_mse = criterion(out['pred'], batch['target'])
            loss = loss_mse + out['loss_hsic'] * 0.1
            loss.backward()
            optimizer.step()
            
    return client_model

def evaluate_model(model, dataloader, device='cpu'):
    model.eval()
    preds_all = []
    targets_all = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch)
            preds_all.append(out['pred'].cpu().numpy())
            targets_all.append(batch['target'].cpu().numpy())
            
    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    return concordance_correlation_coefficient(targets_all, preds_all)

def simulate_federated_training(global_model, train_client_datasets, test_dataset, num_rounds=10, local_epochs=5, device='cpu'):
    global_model.to(device)
    num_clients = len(train_client_datasets)
    train_dataloaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in train_client_datasets]
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    history = []
    
    for round_idx in range(num_rounds):
        client_models = [copy.deepcopy(global_model).to(device) for _ in range(num_clients)]
        
        for i in range(num_clients):
            client_models[i] = client_update(client_models[i], train_dataloaders[i], local_epochs, device)
            
        global_model = fedavg_aggregate(global_model, client_models)
        ccc_score = evaluate_model(global_model, test_dataloader, device)
        
        print(f"Round {round_idx+1}/{num_rounds} FedAvg complete. Test CCC: {ccc_score:.4f}")
        history.append(ccc_score)
        
    return global_model, history
