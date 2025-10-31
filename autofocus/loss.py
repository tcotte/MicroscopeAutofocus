import torch
import torch.nn as nn


class SampleWeightsLoss(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self._beta = beta

    @staticmethod
    def compute_distance_wise_weights(gt_distances: torch.Tensor) -> torch.Tensor:
        # w_i = 1 / (log(|d| + 1) + 1)
        return 1.0 / (torch.log(torch.abs(gt_distances) + 1.0) + 1.0)

    def forward(self, input: torch.Tensor, target: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        input: predicted distances (N,)
        target: ground-truth distances (N,)
        std: feature-based weights (N,)
        """
        diff = input - target
        abs_diff = torch.abs(diff)

        # smooth L1 (Huber) elementwise
        loss_i = torch.where(
            abs_diff < self._beta,
            0.5 * diff ** 2,
            abs_diff - 0.5 * self._beta
        )

        # distance weights
        w = self.compute_distance_wise_weights(target)

        # numerator and denominator
        numerator = torch.sum(w * std * loss_i)
        denominator = torch.sum(w * std) + 1e-8  # avoid div by zero

        loss = numerator / denominator
        return loss
