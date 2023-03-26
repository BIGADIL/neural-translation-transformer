import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

from dataload.data_loaders import *
from enums_and_constants import constants
from models.transformer import Transformer
from tokenizer.bpe_tokenizer import load_bpe_tokenizers
from word2vec.w2v_model import load_w2v_models


def train_model(prune=False) -> None:
    """ Train transformer.

    Parameters
    ----------
        prune: use head pruning while model trains.

    Returns
    -------
        None.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(
        path=constants.DATASET_PATH
    )
    src_tokenizer, trg_tokenizer = load_bpe_tokenizers(
        src_path=constants.SRC_TOKENIZER_PATH,
        trg_path=constants.TRG_TOKENIZER_PATH
    )
    train_data, valid_data, test_data = get_split_datasets(
        dataset=data
    )
    src_w2v, trg_w2v = load_w2v_models(
        src_path=constants.SRC_W2V_PATH,
        trg_path=constants.TRG_W2V_PATH,
        device=device
    )
    model = Transformer(
        src_w2v=src_w2v,
        trg_w2v=trg_w2v,
        model_dim=constants.MODEL_DIM,
        output_dim=trg_tokenizer.get_vocab_size(),
        num_enc_dec_layers=constants.NUM_ENC_DEC_LAYERS,
        heads=constants.HEADS,
        max_seq_len=constants.MAX_SEQ_LEN,
        use_gate=prune,
        eta=constants.ETA
    )

    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param, -0.08, 0.08)

    model.apply(init_weights)

    train_dataloader, valid_dataloader = get_dataloaders(
        datasets=(train_data, valid_data),
        src_tokenizer=src_tokenizer,
        trg_tokenizer=trg_tokenizer,
        device=device
    )

    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=1,
        dirpath=constants.PRUNE_MODEL_CHKPT_PATH if prune else constants.FULL_MODEL_CHKPT_PATH,
        filename="model",
        monitor="val_loss",
        save_top_k=1,
        mode="min"
    )
    es = EarlyStopping(
        monitor="val_loss",
        patience=2,
        min_delta=1e-4,
        verbose=True
    )
    logger = TensorBoardLogger(
        save_dir=constants.PRUNE_MODEL_TRAIN_LOGS if prune else constants.FULL_MODEL_TRAIN_LOGS,
        name="train",
        version="data"
    )
    trainer = pl.Trainer(
        max_epochs=10,
        gpus=-1 if torch.cuda.is_available() else 0,
        callbacks=[
            es,
            checkpoint_callback,
            StochasticWeightAveraging(swa_lrs=1e-2)
        ],
        weights_summary="full",
        gradient_clip_val=0.1,
        track_grad_norm=2,
        gradient_clip_algorithm="norm",
        logger=logger
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prune", required=True, type=bool, help="Use heads pruning")
    args = parser.parse_args()
    train_model(args.prune)
