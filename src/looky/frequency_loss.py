import torch
import torch.nn as nn

class FrequencyLoss(nn.Module):

    def __init__(self, reduction: str = "mean"):
        super().__init__()

        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.fft.fft2(pred, norm="ortho")
        target = torch.fft.fft2(target, norm="ortho")

        dist = (pred - target).abs().pow(2)
        
        if self.reduction == "mean":
            return dist.mean()
        elif self.reduction == "sum":
            return dist.sum()

        return dist
