import math
from typing import Tuple, List

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from torchtext.data import Field
from torchtext.data import TabularDataset


def load_dataset(path: str) -> List:
    src = Field(
        tokenize=lambda x: "<sos>" + x + "<eos>",
        lower=True
    )
    trg = Field(
        tokenize=lambda x: "<sos>" + x + "<eos>",
        lower=True
    )
    dataset = TabularDataset(
        path=path,
        format='tsv',
        fields=[('trg', trg), ('src', src)]
    )
    return dataset.examples


def get_split_datasets(dataset: List, split_ratio: Tuple = (0.7, 0.2, 0.1)) -> Tuple[List, List, List]:
    np.random.seed(17)
    train_end_idx = int(split_ratio[0] * len(dataset))
    val_end_idx = int(sum(split_ratio[:-1]) * len(dataset))
    return dataset[:train_end_idx], dataset[train_end_idx:val_end_idx], dataset[val_end_idx:]


def get_data_loader(dataset: List,
                    sort_function: callable,
                    src_tokenizer: Tokenizer,
                    trg_tokenizer: Tokenizer,
                    device: torch.device,
                    batch_size: int) -> DataLoader:
    res = sorted(dataset, key=sort_function)

    class CustomDataset(Dataset):
        def __init__(self, examples):
            super().__init__()
            self.examples = examples
            self.batch_size = batch_size

        def __getitem__(self, item):
            return self.examples[item * self.batch_size:(item + 1) * self.batch_size]

        def __len__(self):
            return math.ceil(len(self.examples) / self.batch_size)

    def collate_fn(batch):
        batch = batch[0]
        src = [x.src for x in batch]
        trg = [x.trg for x in batch]
        src = src_tokenizer.encode_batch(src)
        trg = trg_tokenizer.encode_batch(trg)
        src = [np.array(x.ids) for x in src]
        trg = [np.array(x.ids) for x in trg]
        src = np.array(src)
        trg = np.array(trg)
        return torch.LongTensor(src).to(device), torch.LongTensor(trg).to(device)

    res = CustomDataset(res)
    res = DataLoader(res, collate_fn=collate_fn)
    return res


def get_dataloaders(datasets: Tuple[List, ...],
                    src_tokenizer: Tokenizer,
                    trg_tokenizer: Tokenizer,
                    device: torch.device,
                    batch_size: int = 64) -> Tuple[DataLoader, ...]:
    def sort_function(x):
        return len(x.src) + len(x.trg)

    res = []
    for dataset in datasets:
        d = get_data_loader(dataset, sort_function, src_tokenizer, trg_tokenizer, device, batch_size)
        res.append(d)
    return tuple(res)
