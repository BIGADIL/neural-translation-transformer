from enum import Enum


class SpecialTokens(Enum):
    UNKNOWN = {"token": "<unk>", "idx": 0}
    START_OF_SEQ = {"token": "<sos>", "idx": 1}
    END_OF_SEQ = {"token": "<eos>", "idx": 2}
    PADDING = {"token": "<pad>", "idx": 3}