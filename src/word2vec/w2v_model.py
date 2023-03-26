import os
from typing import Tuple

import numpy as np
import torch
from gensim.models import Word2Vec
from torch import FloatTensor, LongTensor, vstack
from torch.utils.data import DataLoader

from enums_and_constants import Mode


class W2VModel:
    """ word2vec model for Transformer.
    """

    def __init__(self,
                 device,
                 vector_size=512,
                 workers=8,
                 min_count=5):
        self.w2v_model = Word2Vec(min_count=min_count,
                                  window=5,
                                  vector_size=vector_size,
                                  sample=6e-5,
                                  alpha=0.03,
                                  min_alpha=0.0007,
                                  negative=20,
                                  workers=workers)
        self.mean = None
        self.std = None
        self.device = device

    def train(self,
              dataloader: DataLoader,
              mode: Mode,
              epochs: int = 30):
        """ Train w2v model.

        Parameters
        ----------
            dataloader: where train data store.
            mode: source or target data we need.
            epochs: num train epochs.
        """

        def collect_fn():
            res = []
            for src, trg in dataloader:
                x = src if mode == Mode.SRC else trg
                x = x.tolist()
                res.extend(x)
            return res

        self.w2v_model.build_vocab(collect_fn())
        self.w2v_model.train(collect_fn(), total_examples=self.w2v_model.corpus_count, epochs=epochs)
        self.w2v_model.init_sims(replace=True)
        self.mean = np.mean(self.w2v_model.wv.vectors, axis=0)
        self.std = np.std(self.w2v_model.wv.vectors, axis=0)

    def save(self,
             path: str):
        """ Save w2v model by path.

        Parameters
        ----------
            path: where store model.
        """
        self.w2v_model.save(path)

    def load(self,
             path: str):
        """ Load w2v model by path.

        Parameters
        ----------
            path: where store model.
        """
        self.w2v_model = Word2Vec.load(path)
        self.mean = np.mean(self.w2v_model.wv.vectors, axis=0)
        self.std = np.std(self.w2v_model.wv.vectors, axis=0)

    def __call__(self,
                 *args,
                 **kwargs) -> FloatTensor:
        return self.get_embeddings_(args[0]).to(self.device)

    def get_embeddings_(self,
                        inp: LongTensor) -> FloatTensor:
        """ Get embedding of sentence store in tensor.

        Parameters
        ----------
            inp: tensor with tokens.

        Returns
        -------
            Tensor with sentence embeddings.
        """
        res = []
        for i in range(inp.shape[1]):
            x = vstack([self.get_vector_(y) for y in inp[:, i]])
            res.append(x.unsqueeze(0))
        return FloatTensor(vstack(res)).permute(1, 0, 2)

    def get_vector_(self,
                    word: LongTensor) -> FloatTensor:
        """ Get embedding of word store in tensor.

        Parameters
        ----------
            word: tensor with word's token.

        Returns
        -------
            Tensor with word embedding.
        """
        if int(word) in self.w2v_model.wv.key_to_index:
            vector = self.w2v_model.wv.get_vector(int(word))
            vector = (vector - self.mean) / self.std
            return FloatTensor(vector)
        return FloatTensor(self.mean)


def load_w2v_models(src_path: str,
                    trg_path: str,
                    device: torch.device) -> Tuple[W2VModel, W2VModel]:
    """ Get embedding of sentence store in tensor.

        Parameters
        ----------
            src_path: path to source w2v model.
            trg_path: path to target w2v model.
            device: where store tensors (cpu or cuda).

        Returns
        -------
            Tuple of source and target w2v models.
        """
    if not os.path.exists(src_path):
        raise Exception("src w2v does not exist")
    if not os.path.exists(trg_path):
        raise Exception("trg w2v does not exist")
    src_w2v = W2VModel(device=device)
    src_w2v.load(path=src_path)
    trg_w2v = W2VModel(device=device)
    trg_w2v.load(path=trg_path)
    return src_w2v, trg_w2v
