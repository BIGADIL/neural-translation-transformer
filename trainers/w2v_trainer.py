import torch
from torch.utils.data import DataLoader

from dataload import load_dataset, get_dataloaders
from enums_and_constants import constants, Mode
from tokenizer import load_bpe_tokenizers
from word2vec import W2VModel


def train_and_save_w2v_model(dataloader: DataLoader,
                             mode: Mode,
                             save_path: str) -> None:
    """ Train tokenizer and save it.

    Parameters
    ----------
        dataloader: dataloader with examples.
        mode: source or target examples should be used.
        save_path: where store w2v model.

    Returns
    -------
        None.
    """
    w2v = W2VModel(device=torch.device("cpu"))
    w2v.train(dataloader=dataloader, mode=mode)
    w2v.save(path=save_path)


if __name__ == '__main__':
    data = load_dataset(path=constants.DATASET_PATH)
    src_tokenizer, trg_tokenizer = load_bpe_tokenizers(
        src_path=constants.SRC_TOKENIZER_PATH,
        trg_path=constants.TRG_TOKENIZER_PATH
    )
    dataloaders = get_dataloaders(
        datasets=(data,),
        src_tokenizer=src_tokenizer,
        trg_tokenizer=trg_tokenizer,
        device=torch.device("cpu")
    )
    train_and_save_w2v_model(
        dataloader=dataloaders[0],
        mode=Mode.SRC,
        save_path=constants.SRC_W2V_PATH
    )
    train_and_save_w2v_model(
        dataloader=dataloaders[0],
        mode=Mode.TRG,
        save_path=constants.TRG_W2V_PATH
    )
