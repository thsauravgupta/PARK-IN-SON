import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

def extract_attention_weights(model, dataloader, device='cpu'):
    model.eval()
    all_attn_mri = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch)
            # Shape is expected to be (N, num_heads, tgt_len, src_len)
            all_attn_mri.append(out['attn_weights']['mri'].cpu().numpy())
            
    return np.concatenate(all_attn_mri, axis=0)

def visualize_attention(attn_weights, save_path="outputs/figures/attention_maps.png"):
    # Flatten the weights to plot appropriately even if it is a single value from sequence length 1
    avg_attn = np.atleast_1d(attn_weights.mean(axis=0).flatten())
    
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(avg_attn)), avg_attn)
    plt.title("Average Phenotype-Guided Attention over MRI Features")
    plt.xlabel("MRI Embedding Dimension")
    plt.ylabel("Attention Weight")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class IGModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, clin, mri, mri_mask, pet, pet_mask, gen, gen_mask):
        batch = {
            'clinical': clin,
            'mri': mri,
            'mri_mask': mri_mask,
            'pet': pet,
            'pet_mask': pet_mask,
            'genetic': gen,
            'genetic_mask': gen_mask
        }
        return self.model(batch)['pred']

def compute_integrated_gradients(model, batch, device='cpu'):
    wrapper = IGModelWrapper(model).to(device)
    ig = IntegratedGradients(wrapper)
    
    inputs = (
        batch['clinical'].requires_grad_(),
        batch['mri'].requires_grad_(),
        batch['mri_mask'],
        batch['pet'].requires_grad_(),
        batch['pet_mask'],
        batch['genetic'].requires_grad_(),
        batch['genetic_mask']
    )
    
    attr = ig.attribute(inputs, target=None)
    return {
        'clinical': attr[0],
        'mri': attr[1],
        'pet': attr[3],
        'genetic': attr[5]
    }

def stress_test_missing_modalities(model, dataloader, device='cpu', save_path="outputs/figures/modality_robustness.png"):
    """Evaluates the model by artificially zeroing out specific modalities to show robustness via learned missing tokens."""
    model.eval()
    
    # Store predictions for different dropout scenarios
    scenarios = {
        'Full Data': {'drop_mri': False, 'drop_pet': False, 'drop_gen': False},
        'Missing MRI': {'drop_mri': True, 'drop_pet': False, 'drop_gen': False},
        'Missing PET': {'drop_mri': False, 'drop_pet': True, 'drop_gen': False},
        'Missing Genetics': {'drop_mri': False, 'drop_pet': False, 'drop_gen': True},
        'Clinical Only': {'drop_mri': True, 'drop_pet': True, 'drop_gen': True}
    }
    
    metrics = {k: 0.0 for k in scenarios.keys()}
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            b_size = batch['clinical'].size(0)
            total_samples += b_size
            
            targets = batch['targets'].to(device).float().squeeze()
            
            for name, config in scenarios.items():
                test_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        test_batch[k] = v.clone().to(device)
                    else:
                        test_batch[k] = v
                
                # Apply forced dropouts
                if config['drop_mri']:
                    test_batch['mri_mask'] = torch.ones_like(test_batch['mri_mask'])
                if config['drop_pet']:
                    test_batch['pet_mask'] = torch.ones_like(test_batch['pet_mask'])
                if config['drop_gen']:
                    test_batch['genetic_mask'] = torch.ones_like(test_batch['genetic_mask'])
                
                preds = model(test_batch)['pred'].squeeze()
                mse = torch.sum((preds - targets) ** 2).item()
                metrics[name] += mse

    plt.figure(figsize=(10, 5))
    names = list(metrics.keys())
    # Calculate RMSE
    rmses = [np.sqrt(metrics[n] / total_samples) for n in names]
    
    colors = ['#2ecc71', '#f1c40f', '#f39c12', '#e67e22', '#e74c3c']
    bars = plt.bar(names, rmses, color=colors)
    plt.title("Fed-PhenoGraft Robustness to Missing Modalities (RMSE)")
    plt.ylabel("Root Mean Squared Error (Lower is Better)")
    plt.ylim(0, max(rmses) * 1.3)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f'{yval:.2f}', va='bottom', ha='center')
        
    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def visualize_feature_importance(model, dataloader, device='cpu', save_path="outputs/figures/global_feature_importance.png"):
    """Computes global feature importance across modalities using Integrated Gradients."""
    # Compute IG for just the first batch to represent global dynamics
    batch = next(iter(dataloader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
    attrs = compute_integrated_gradients(model, batch, device=device)
    
    # Take absolute mean across batch dimension and feature dimension for a coarse summary
    global_importance = {
        'Clinical/Demographic': np.abs(attrs['clinical'].cpu().detach().numpy()).mean(),
        'Structural MRI': np.abs(attrs['mri'].cpu().detach().numpy()).mean(),
        'DaTScan PET': np.abs(attrs['pet'].cpu().detach().numpy()).mean(),
        'Genetics (Carriers)': np.abs(attrs['genetic'].cpu().detach().numpy()).mean()
    }
    
    modalities = list(global_importance.keys())
    importances = list(global_importance.values())
    
    # Normalize
    total = sum(importances) + 1e-8
    importances = [i / total * 100 for i in importances]
    
    plt.figure(figsize=(8, 8))
    plt.pie(importances, labels=modalities, autopct='%1.1f%%', startangle=140, colors=['#3498db', '#9b59b6', '#e74c3c', '#2ecc71'])
    plt.title("Global Multimodal Feature Attribution (Integrated Gradients)")
    
    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

