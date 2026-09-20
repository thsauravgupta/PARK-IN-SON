import torch
import torch.nn as nn


class CCCLoss(nn.Module):
    """
    Concordance Correlation Coefficient loss.

    CCC measures both:
        1. correlation
        2. agreement between prediction and target

    Loss:
        1 - CCC

    Higher CCC is better.
    Lower CCC loss is better.
    """

    def __init__(self, eps=1e-8):
        super().__init__()

        self.eps = eps

    def forward(self, pred, target):
        """
        Parameters
        ----------
        pred : torch.Tensor
            Model predictions.

        target : torch.Tensor
            Ground-truth targets.

        Returns
        -------
        torch.Tensor
            CCC loss = 1 - CCC.
        """

        pred = pred.reshape(-1)
        target = target.reshape(-1)

        if pred.numel() < 2:
            return torch.tensor(
                0.0,
                device=pred.device,
                dtype=pred.dtype,
            )

        pred_mean = pred.mean()
        target_mean = target.mean()

        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        covariance = (
            pred_centered * target_centered
        ).mean()

        pred_variance = (
            pred_centered ** 2
        ).mean()

        target_variance = (
            target_centered ** 2
        ).mean()

        numerator = (
            2.0 * covariance
        )

        denominator = (
            pred_variance
            + target_variance
            + (pred_mean - target_mean) ** 2
            + self.eps
        )

        ccc = numerator / denominator

        return 1.0 - ccc