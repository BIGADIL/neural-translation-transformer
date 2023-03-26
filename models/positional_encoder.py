import math

import torch
import torch.nn as nn
from torch import FloatTensor
from torch.autograd import Variable


class PositionalEncoder(nn.Module):
    """ PositionalEncoder of Transformer.
    Original code: https://github.com/SamLynnEvans/Transformer
    """

    def __init__(self,
                 model_dim: int,
                 max_seq_len: int):
        super().__init__()
        self.model_dim = model_dim

        pe = torch.zeros(max_seq_len, model_dim)
        for pos in range(max_seq_len):
            for i in range(0, model_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / model_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / model_dim)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self,
                x: FloatTensor) -> FloatTensor:
        # make embeddings relatively larger
        x = x * math.sqrt(self.model_dim)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False)
        return x
