import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor


class FeedForward(nn.Module):
    """ FeedForward layer of Transformer.
    Original code: https://github.com/SamLynnEvans/Transformer
    Use SiLU instead of ReLU.
    """

    def __init__(self,
                 model_dim: int,
                 d_ff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(model_dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, model_dim)

    def forward(self,
                x: FloatTensor) -> FloatTensor:
        x = self.dropout(F.silu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
