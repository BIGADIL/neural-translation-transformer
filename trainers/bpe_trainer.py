from typing import List

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from torchtext.data import Example

from dataload.data_loaders import load_dataset
from enums_and_constants import constants
from enums_and_constants.mode import Mode
from enums_and_constants.special_tokens import SpecialTokens


def train_tokenizer(dataset: List[Example],
                    mode: Mode,
                    save_path: str):
    """ Train tokenizer and save it.

    Parameters
    ----------
        dataset: list of source and target examples.
        mode: source or target examples should be used.
        save_path: where store tokenizer.

    Returns
    -------
        None.
    """

    def iterator():
        for example in dataset:
            if mode == Mode.SRC:
                yield example.src
            else:
                yield example.trg

    tokenizer = Tokenizer(BPE(unk_token=SpecialTokens.UNKNOWN.value['token']))
    pad = SpecialTokens.PADDING.value
    special_tokens = [x.value['token'] for x in SpecialTokens]
    trainer = BpeTrainer(special_tokens=special_tokens)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(iterator(), trainer=trainer)
    tokenizer.enable_padding(pad_id=pad['idx'], pad_token=pad['token'])
    tokenizer.save(save_path)


if __name__ == '__main__':
    data = load_dataset(path=constants.DATASET_PATH)
    train_tokenizer(dataset=data, mode=Mode.SRC, save_path=constants.SRC_TOKENIZER_PATH)
    train_tokenizer(dataset=data, mode=Mode.TRG, save_path=constants.TRG_TOKENIZER_PATH)
