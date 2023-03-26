from typing import Tuple, Union

import torch.nn as nn
from torch import LongTensor, BoolTensor, FloatTensor

from models import FeedForward, MultiHeadAttention, Norm, PositionalEncoder
from word2vec import W2VModel


class EncoderLayer(nn.Module):
    """ Encoder layer of Transformer.
    Original code: https://github.com/SamLynnEvans/Transformer
    """

    def __init__(self,
                 model_dim: int,
                 heads: int,
                 use_gate: bool,
                 dropout: float = 0.1):
        super().__init__()
        self.norm_1 = Norm(model_dim)
        self.norm_2 = Norm(model_dim)
        self.attn = MultiHeadAttention(heads, model_dim, use_gate)
        self.ff = FeedForward(model_dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def get_infer_gate_info(self):
        return [["attn", self.attn.get_infer_gate_info()]]

    def forward(self, x, mask) -> Tuple[FloatTensor, Union[float, FloatTensor]]:
        x2 = self.norm_1(x)
        a, l0_loss = self.attn(x2, x2, x2, mask)
        x = x + self.dropout_1(a)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x, l0_loss


class Encoder(nn.Module):
    """ Encoder of Transformer.
    Original code: https://github.com/SamLynnEvans/Transformer
    """

    def __init__(self,
                 w2v: W2VModel,
                 model_dim: int,
                 num_encoders: int,
                 heads: int,
                 max_seq_len: int,
                 use_gate: bool):
        super().__init__()
        self.w2v = w2v
        self.num_encoders = num_encoders
        self.pe = PositionalEncoder(model_dim, max_seq_len)
        self.layers = nn.ModuleList([EncoderLayer(model_dim, heads, use_gate) for _ in range(num_encoders)])
        self.norm = Norm(model_dim)

    def get_infer_gate_info(self):
        result = []
        for i in range(self.num_encoders):
            igi = self.layers[i].get_infer_gate_info()
            for j in range(len(igi)):
                igi[j][0] = f"encoder_layer_{i}." + igi[j][0]
            result.extend(igi)
        return result

    def forward(self,
                src: LongTensor,
                mask: BoolTensor) -> Tuple[FloatTensor, FloatTensor]:
        x = self.w2v(src)
        x = self.pe(x)
        l0_loss = FloatTensor([0]).to(src.device)
        for i in range(self.num_encoders):
            x, l0_loss_layer = self.layers[i](x, mask)
            l0_loss += l0_loss_layer
        return self.norm(x), l0_loss
