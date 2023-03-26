import os.path
from typing import Tuple

from tokenizers import Tokenizer
from tokenizers.models import BPE

from enums_and_constants import SpecialTokens


def load_bpe_tokenizers(src_path: str,
                        trg_path: str) -> Tuple[Tokenizer, Tokenizer]:
    """ Load source and target BPE-tokenizers from data folder.

    Parameters
    ----------
        src_path: path to source tokenizer.
        trg_path: path to target tokenizer.

    Returns
    -------
        Tuple of source and target tokenizers.
    """
    if not os.path.exists(src_path):
        raise Exception("src tokenizer does not exist")
    if not os.path.exists(trg_path):
        raise Exception("trg tokenizer does not exist")
    src_tokenizer = Tokenizer(model=BPE(unk_token=SpecialTokens.UNKNOWN.value['token']))
    pad = SpecialTokens.PADDING.value
    src_tokenizer = src_tokenizer.from_file(path=src_path)
    src_tokenizer.enable_padding(pad_id=pad['idx'], pad_token=pad['token'])

    trg_tokenizer = Tokenizer(model=BPE(unk_token=SpecialTokens.UNKNOWN.value['token']))
    pad = SpecialTokens.PADDING.value
    trg_tokenizer = trg_tokenizer.from_file(path=trg_path)
    trg_tokenizer.enable_padding(pad_id=pad['idx'], pad_token=pad['token'])

    return src_tokenizer, trg_tokenizer
