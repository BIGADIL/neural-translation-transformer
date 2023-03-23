import torch
import torch.nn as nn
from torch import FloatTensor


class Norm(nn.Module):
    def __init__(self,
                 model_dim: int,
                 eps: float = 1e-6):
        super().__init__()
        self.model_dim = model_dim
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.model_dim))
        self.bias = nn.Parameter(torch.zeros(self.model_dim))
        self.eps = eps

    def forward(self,
                x: FloatTensor) -> FloatTensor:
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
