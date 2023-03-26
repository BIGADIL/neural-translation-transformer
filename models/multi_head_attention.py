import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, BoolTensor, Tensor

from models.gate import Gate


class MultiHeadAttention(nn.Module):
    """ MultiHeadAttention layer of Transformer.
    Original code: https://github.com/SamLynnEvans/Transformer
    """

    def __init__(self,
                 heads: int,
                 model_dim: int,
                 use_gate: bool,
                 dropout: float = 0.1):
        super().__init__()
        self.model_dim = model_dim
        self.head_dim = model_dim // heads
        self.heads = heads

        self.use_gate = use_gate
        self.gate = Gate(heads)

        self.q_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(model_dim, model_dim)

    def get_infer_gate_info(self):
        return self.gate()[0].flatten().tolist()

    def forward(self,
                q: FloatTensor,
                k: FloatTensor,
                v: FloatTensor,
                mask: Optional[BoolTensor] = None) -> Tuple[FloatTensor, FloatTensor]:
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.heads, self.head_dim)
        q = self.q_linear(q).view(bs, -1, self.heads, self.head_dim)
        v = self.v_linear(v).view(bs, -1, self.heads, self.head_dim)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = MultiHeadAttention.attention(q, k, v, self.head_dim, mask, self.dropout)
        l0_loss = 0
        if self.use_gate:
            gate, l0_loss = self.gate()
            scores = scores * gate[None, :, None, None]

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.model_dim)

        output = self.out(concat)

        return output, l0_loss

    @staticmethod
    def attention(
            q: FloatTensor,
            k: FloatTensor,
            v: FloatTensor,
            head_dim: int,
            mask: Optional[BoolTensor] = None,
            dropout: Optional[nn.Dropout] = None) -> Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        return output
