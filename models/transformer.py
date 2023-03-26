from typing import Union, List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT, EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import LongTensor, BoolTensor
from torch.autograd import Variable

from enums_and_constants.special_tokens import SpecialTokens
from models.decoder import Decoder
from models.encoder import Encoder
from word2vec.w2v_model import W2VModel


class Transformer(pl.LightningModule):
    """ Transformer.
    Original code: https://github.com/SamLynnEvans/Transformer
    """

    def __init__(self,
                 src_w2v: W2VModel,
                 trg_w2v: W2VModel,
                 model_dim: int,
                 output_dim: int,
                 num_enc_dec_layers: int,
                 heads: int,
                 max_seq_len: int,
                 use_gate: bool,
                 eta: float):
        super().__init__()
        self.encoder = Encoder(src_w2v, model_dim, num_enc_dec_layers, heads, max_seq_len, use_gate)
        self.decoder = Decoder(trg_w2v, model_dim, num_enc_dec_layers, heads, max_seq_len, use_gate)
        self.out = nn.Linear(model_dim, output_dim)
        self.loss = nn.CrossEntropyLoss(ignore_index=SpecialTokens.PADDING.value['idx'])
        self.eta = eta

    def get_infer_gate_info(self):
        result = []
        igi = self.encoder.get_infer_gate_info()
        for i in range(len(igi)):
            igi[i][0] = "encoder." + igi[i][0]
        result.extend(igi)
        igi = self.decoder.get_infer_gate_info()
        for i in range(len(igi)):
            igi[i][0] = "decoder." + igi[i][0]
        result.extend(igi)
        return result

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs, l0_loss_e = self.encoder(src, src_mask)
        d_output, l0_loss_d = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output, l0_loss_e + l0_loss_d

    def configure_optimizers(self):
        learning_rate = 1e-3
        optimiser = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        monitor = "train_loss"
        return {
            'optimizer': optimiser,
            'lr_scheduler': torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.6),
            'monitor': monitor
        }

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        src, trg = args[0]
        src_mask = Transformer.src_mask(src)
        transformer_loss = torch.FloatTensor([0]).to(self.device)
        l0_loss = torch.FloatTensor([0]).to(self.device)
        for i in range(1, trg.shape[1]):
            trg_input = trg[:, :i]
            target = trg[:, 1:i + 1]
            trg_mask = Transformer.trg_mask(trg_input, self.device)
            preds, l0_slice = self(src, trg_input, src_mask, trg_mask)
            preds = preds.reshape(-1, preds.size(2))
            target = target.reshape(-1)
            transformer_loss += self.loss(preds, target)
            l0_loss += l0_slice
        transformer_loss /= trg.shape[1] - 1
        l0_loss /= trg.shape[1] - 1
        self.log("transformer_loss", float(transformer_loss), prog_bar=True)
        self.log("l0_loss", float(l0_loss), prog_bar=True)
        self.log("loss", float(transformer_loss + self.eta * l0_loss), prog_bar=True)
        result = {"loss": transformer_loss + self.eta * l0_loss}
        return result

    def validation_step(self, *args, **kwargs) -> STEP_OUTPUT:
        src, trg = args[0]
        src_mask = Transformer.src_mask(src)
        transformer_loss = torch.FloatTensor([0]).to(self.device)
        l0_loss = torch.FloatTensor([0]).to(self.device)
        for i in range(1, trg.shape[1]):
            trg_input = trg[:, :i]
            target = trg[:, 1:i + 1]
            trg_mask = Transformer.trg_mask(trg_input, self.device)
            preds, l0_slice = self(src, trg_input, src_mask, trg_mask)
            preds = preds.reshape(-1, preds.size(2))
            target = target.reshape(-1)
            transformer_loss += self.loss(preds, target)
            l0_loss += l0_slice
        transformer_loss /= trg.shape[1] - 1
        l0_loss /= trg.shape[1] - 1
        self.log("val_transformer_loss", float(transformer_loss), prog_bar=True)
        self.log("val_l0_loss", float(l0_loss), prog_bar=True)
        self.log("val_loss", float(transformer_loss + self.eta * l0_loss), prog_bar=True)
        result = {"val_loss": transformer_loss + self.eta * l0_loss}
        return result

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        losses = [i['loss'] for i in outputs]
        self.log("train_loss_end", float((sum(losses) / len(losses)).detach().cpu()), prog_bar=True)

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        losses = [i['val_loss'] for i in outputs]
        self.log("val_loss_end", float((sum(losses) / len(losses)).detach().cpu()), prog_bar=True)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    @staticmethod
    def src_mask(src: LongTensor) -> BoolTensor:
        pti = SpecialTokens.PADDING.value['idx']
        mask = (src != pti).unsqueeze(1)
        return mask

    @staticmethod
    def trg_mask(trg: LongTensor,
                 device: torch.device) -> BoolTensor:
        pti = SpecialTokens.PADDING.value['idx']
        mask = (trg != pti).unsqueeze(1)
        size = trg.size(1)
        no_peak_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
        no_peak_mask = Variable(torch.from_numpy(no_peak_mask) == 0).to(device)
        mask = mask & no_peak_mask
        return mask
