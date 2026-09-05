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
            
            targets = batch['target'].to(device).float().squeeze()
            
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


def counterfactual_gene_analysis(model, preprocessor, clin_raw, mri_raw, pet_raw,
                                 gen_raw, device='cpu',
                                 save_path="outputs/figures/counterfactual_genes.png",
                                 target_label="UPDRS-III"):
    """
    Counterfactual "what-if" analysis on genetic carrier status.

    For each PD-risk gene, predicts the Year-2 UPDRS-III for every test subject
    twice — once with carrier status forced to 0 and once forced to 1 (derived
    columns n_variants / lrrk2_positive / gba_positive are recomputed so the
    counterfactual is internally consistent). Flips happen in RAW feature space
    and are re-run through the train-fit preprocessor, so scaled values stay
    exact. The bar chart shows the mean predicted progression shift per gene.
    """
    import os
    import pandas as pd
    from src.data.dataset import FederatedPPMIDataset
    from torch.utils.data import DataLoader

    genes = [g for g in ["LRRK2", "GBA", "SNCA", "PINK1", "PRKN", "APOE_e4_carrier"]
             if g in gen_raw.columns]
    if not genes:
        return None

    def _consistent(gen_df):
        gen_df = gen_df.copy()
        carrier_cols = [c for c in ["LRRK2", "GBA", "SNCA", "PINK1", "PRKN"]
                        if c in gen_df.columns]
        if "n_variants" in gen_df.columns and carrier_cols:
            gen_df["n_variants"] = gen_df[carrier_cols].sum(axis=1)
        if "lrrk2_positive" in gen_df.columns and "LRRK2" in gen_df.columns:
            gen_df["lrrk2_positive"] = (gen_df["LRRK2"] > 0).astype(float)
        if "gba_positive" in gen_df.columns and "GBA" in gen_df.columns:
            gen_df["gba_positive"] = (gen_df["GBA"] > 0).astype(float)
        return gen_df

    def _predict(gen_df):
        c, m, p, g = preprocessor.transform(clin_raw, mri_raw, pet_raw, gen_df)
        ds = FederatedPPMIDataset(c, m, p, g,
                                  pd.Series(np.zeros(len(c)), index=c.index))
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        preds = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                preds.append(model(batch)['pred'].cpu().numpy())
        return np.concatenate(preds)

    deltas = {}
    for gene in genes:
        gen_neg = gen_raw.copy(); gen_neg[gene] = 0.0
        gen_pos = gen_raw.copy(); gen_pos[gene] = 1.0
        preds_neg = _predict(_consistent(gen_neg))
        preds_pos = _predict(_consistent(gen_pos))
        deltas[gene] = float(np.mean(preds_pos - preds_neg))

    plt.figure(figsize=(9, 5))
    names = list(deltas.keys())
    values = [deltas[n] for n in names]
    colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in values]
    bars = plt.bar(names, values, color=colors)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(f"Counterfactual: Predicted {target_label} Shift if Carrier vs Non-Carrier")
    plt.ylabel(f"Mean Δ Predicted {target_label} (points)")
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, v,
                 f'{v:+.2f}', va='bottom' if v >= 0 else 'top', ha='center')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    return deltas


def plot_pred_vs_actual(model, dataloader, device='cpu',
                        save_path="outputs/figures/pred_vs_actual.png",
                        target_label="UPDRS-III @ Year 2"):
    """Scatter of predicted vs actual regression target on the held-out test set."""
    import os
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            preds.append(model(batch)['pred'].cpu().numpy())
            targets.append(batch['target'].cpu().numpy())
    preds, targets = np.concatenate(preds), np.concatenate(targets)

    lim = [min(targets.min(), preds.min()) - 2, max(targets.max(), preds.max()) + 2]
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.4, s=18, color='#3498db', edgecolors='none')
    plt.plot(lim, lim, 'r--', linewidth=1, label='Perfect agreement')
    plt.xlim(lim); plt.ylim(lim)
    plt.xlabel(f"Actual {target_label}")
    plt.ylabel(f"Predicted {target_label}")
    plt.title("Fed-PhenoGraft: Predicted vs Actual (Held-out Test)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(model, dataloader, device='cpu',
                          save_path="outputs/figures/confusion_matrix.png"):
    """Confusion matrix for the PD vs HC classification head on the test set."""
    import os
    from sklearn.metrics import confusion_matrix
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            logits.append(model(batch)['cls_logit'].cpu().numpy())
            labels.append(batch['diagnosis'].cpu().numpy())
    logits, labels = np.concatenate(logits), np.concatenate(labels)
    if len(np.unique(labels)) < 2:
        return None
    preds = (1.0 / (1.0 + np.exp(-logits)) > 0.5).astype(int)

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    ticks = ['HC / Other (0)', 'PD (1)']
    plt.xticks([0, 1], ticks); plt.yticks([0, 1], ticks)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("PD vs HC Classification (Held-out Test)")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    return cm.tolist()


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

