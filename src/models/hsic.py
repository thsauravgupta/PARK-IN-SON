import torch
import torch.nn as nn

class HSICLoss(nn.Module):
    """
    Hilbert-Schmidt Independence Criterion to orthogonalize representations.
    Forces the 'shared' and 'private' latents to be statistically independent.
    """
    def __init__(self, sigma=1.0):
        super(HSICLoss, self).__init__()
        self.sigma = sigma

    def rbf_kernel(self, X, Y):
        # X: N x d
        # Y: N x d
        dist = torch.cdist(X, Y, p=2) ** 2
        return torch.exp(-dist / (2 * self.sigma ** 2))

    def forward(self, z_shared, z_private):
        N = z_shared.size(0)
        if N < 2:
            return torch.tensor(0.0).to(z_shared.device)
            
        K = self.rbf_kernel(z_shared, z_shared)
        L = self.rbf_kernel(z_private, z_private)
        
        H = torch.eye(N).to(z_shared.device) - (1.0/N) * torch.ones(N, N).to(z_shared.device)
        
        Kc = torch.mm(K, H)
        Lc = torch.mm(L, H)
        
        hsic = torch.trace(torch.mm(Kc, Lc)) / ((N - 1)**2)
        return hsic
