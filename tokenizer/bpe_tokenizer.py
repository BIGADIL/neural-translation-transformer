from typing import Tuple

from tokenizers import Tokenizer
from tokenizers.models import BPE

from enums_and_constants.special_tokens import SpecialTokens


def load_bpe_tokenizers(src_path: str,
                        trg_path: str) -> Tuple[Tokenizer, Tokenizer]:
    src_tokenizer = Tokenizer(model=BPE(unk_token=SpecialTokens.UNKNOWN.value['token']))
    pad = SpecialTokens.PADDING.value
    src_tokenizer = src_tokenizer.from_file(path=src_path)
    src_tokenizer.enable_padding(pad_id=pad['idx'], pad_token=pad['token'])

    trg_tokenizer = Tokenizer(model=BPE(unk_token=SpecialTokens.UNKNOWN.value['token']))
    pad = SpecialTokens.PADDING.value
    trg_tokenizer = trg_tokenizer.from_file(path=trg_path)
    trg_tokenizer.enable_padding(pad_id=pad['idx'], pad_token=pad['token'])

    return src_tokenizer, trg_tokenizer
