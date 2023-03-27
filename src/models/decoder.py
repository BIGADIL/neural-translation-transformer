from typing import Tuple

import torch.nn as nn
from torch import FloatTensor, LongTensor, BoolTensor

from models import FeedForward, MultiHeadAttention, Norm, PositionalEncoder
from word2vec import W2VModel


class DecoderLayer(nn.Module):
    """ Decoder layer of Transformer.
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
        self.norm_3 = Norm(model_dim)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, model_dim, use_gate)
        self.attn_2 = MultiHeadAttention(heads, model_dim, use_gate)
        self.ff = FeedForward(model_dim)

    def get_infer_gate_info(self):
        return [["attn_1", self.attn_1.get_infer_gate_info()], ["attn_2", self.attn_2.get_infer_gate_info()]]

    def forward(self,
                x: FloatTensor,
                e_outputs: FloatTensor,
                src_mask: BoolTensor,
                trg_mask: BoolTensor) -> Tuple[FloatTensor, FloatTensor]:
        x2 = self.norm_1(x)
        a_1, l0_loss_1 = self.attn_1(x2, x2, x2, trg_mask)
        x = x + self.dropout_1(a_1)
        x2 = self.norm_2(x)
        a_2, l0_loss_2 = self.attn_2(x2, e_outputs, e_outputs, src_mask)
        x = x + self.dropout_2(a_2)
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x, l0_loss_1 + l0_loss_2


class Decoder(nn.Module):
    """ Decoder of Transformer.
    Original code: https://github.com/SamLynnEvans/Transformer
    """

    def __init__(self,
                 w2v: W2VModel,
                 model_dim: int,
                 num_decoders: int,
                 heads: int,
                 max_seq_len: int,
                 use_gate: bool):
        super().__init__()
        self.num_decoders = num_decoders
        self.w2v = w2v
        self.pe = PositionalEncoder(model_dim, max_seq_len)
        self.layers = nn.ModuleList([DecoderLayer(model_dim, heads, use_gate) for _ in range(num_decoders)])
        self.norm = Norm(model_dim)

    def get_infer_gate_info(self):
        result = []
        for i in range(self.num_decoders):
            igi = self.layers[i].get_infer_gate_info()
            for j in range(len(igi)):
                igi[j][0] = f"decoder_layer_{i}." + igi[j][0]
            result.extend(igi)
        return result

    def forward(self,
                trg: LongTensor,
                e_outputs: FloatTensor,
                src_mask: BoolTensor,
                trg_mask: BoolTensor) -> Tuple[FloatTensor, FloatTensor]:
        x = self.w2v(trg)
        x = self.pe(x)
        l0_loss = FloatTensor([0]).to(trg.device)
        for i in range(self.num_decoders):
            x, l0_loss_layer = self.layers[i](x, e_outputs, src_mask, trg_mask)
            l0_loss += l0_loss_layer
        return self.norm(x), l0_loss
