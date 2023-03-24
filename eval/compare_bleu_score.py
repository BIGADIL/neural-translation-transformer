import os

import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu
from tokenizers import Tokenizer
from torch.autograd import Variable
from tqdm import tqdm

from dataload.data_loaders import load_dataset, get_split_datasets, get_dataloaders
from enums_and_constants import constants
from enums_and_constants.special_tokens import SpecialTokens
from models.transformer import Transformer
from preload.preloader import preload_data_from_gdrive
from tokenizer.bpe_tokenizer import load_bpe_tokenizers
from word2vec.w2v_model import load_w2v_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_batch(batch,
                  model: Transformer,
                  trg_tokenizer: Tokenizer):
    eti = SpecialTokens.END_OF_SEQ.value['idx']

    def is_in(row):
        return eti in row

    src, trg = batch
    src_mask = Transformer.src_mask(src)
    e_outputs, _ = model.encoder(src, src_mask)
    outs = torch.zeros(src.size(0), constants.MAX_SEQ_LEN).type_as(src.data).fill_(SpecialTokens.PADDING.value['idx'])
    outs[:, 0] = SpecialTokens.START_OF_SEQ.value['idx']
    for i in range(1, constants.MAX_SEQ_LEN):
        trg_mask = np.triu(np.ones((1, i, i)), k=1).astype('uint8')
        trg_mask = Variable(torch.from_numpy(trg_mask) == 0).to(device)
        preds = model.out(model.decoder(outs[:, :i], e_outputs, src_mask, trg_mask)[0])
        idx = preds[:, -1, :].argmax(dim=1)
        outs[:, i] = idx
        check = outs.cpu().numpy()
        if np.all(np.apply_along_axis(is_in, 1, check)):
            break
    trg = trg.tolist()
    outs = outs.tolist()
    for i in range(len(outs)):
        if eti in outs[i]:
            idx = outs[i].index(eti)
            outs[i] = outs[i][:idx + 1]
    trg = trg_tokenizer.decode_batch(trg)
    outs = trg_tokenizer.decode_batch(outs)
    trg = [x.tokens for x in trg_tokenizer.encode_batch(trg)]
    outs = [x.tokens for x in trg_tokenizer.encode_batch(outs)]
    trg = [[y for y in x if y != SpecialTokens.PADDING.value['token']] for x in trg]
    outs = [[y for y in x if y != SpecialTokens.PADDING.value['token']] for x in outs]
    return trg, outs


def compute_bleu():
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
    full_model = Transformer(
        src_w2v=src_w2v,
        trg_w2v=trg_w2v,
        model_dim=constants.MODEL_DIM,
        output_dim=trg_tokenizer.get_vocab_size(),
        num_enc_dec_layers=constants.NUM_ENC_DEC_LAYERS,
        heads=constants.HEADS,
        max_seq_len=constants.MAX_SEQ_LEN,
        use_gate=False,
        eta=constants.ETA
    )
    prune_model = Transformer(
        src_w2v=src_w2v,
        trg_w2v=trg_w2v,
        model_dim=constants.MODEL_DIM,
        output_dim=trg_tokenizer.get_vocab_size(),
        num_enc_dec_layers=constants.NUM_ENC_DEC_LAYERS,
        heads=constants.HEADS,
        max_seq_len=constants.MAX_SEQ_LEN,
        use_gate=True,
        eta=constants.ETA
    )
    state_dict = torch.load(os.path.join(constants.FULL_MODEL_CHKPT_PATH, "model.ckpt"))["state_dict"]
    full_model.load_state_dict(state_dict)
    full_model.to(device)
    full_model.eval()

    state_dict = torch.load(os.path.join(constants.PRUNE_MODEL_CHKPT_PATH, "model.ckpt"))["state_dict"]
    prune_model.load_state_dict(state_dict)
    prune_model.to(device)
    prune_model.eval()
    infer_gate_info = prune_model.get_infer_gate_info()
    total_heads = 0
    alive_heads = 0
    for igi in infer_gate_info:
        print(igi[0], ":", igi[1])
        total_heads += len(igi[1])
        alive_heads += sum(igi[1])
    print("total heads: ", total_heads)
    print("alive heads", int(alive_heads))

    original_text_full_model = []
    generated_text_full_model = []
    original_text_prune_model = []
    generated_text_prune_model = []

    test_dataloader, = get_dataloaders(
        datasets=(test_data,),
        src_tokenizer=src_tokenizer,
        trg_tokenizer=trg_tokenizer,
        device=device,
        batch_size=128
    )

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            ot, gt = process_batch(batch, full_model, trg_tokenizer)
            original_text_full_model.extend(ot)
            generated_text_full_model.extend(gt)
            ot, gt = process_batch(batch, prune_model, trg_tokenizer)
            original_text_prune_model.extend(ot)
            generated_text_prune_model.extend(gt)

    print(corpus_bleu([[text] for text in original_text_full_model], generated_text_full_model) * 100)
    print(corpus_bleu([[text] for text in original_text_prune_model], generated_text_prune_model) * 100)


if __name__ == '__main__':
    preload_data_from_gdrive()
    compute_bleu()
