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
