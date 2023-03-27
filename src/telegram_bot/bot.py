import os

import language_tool_python
import numpy as np
import telebot
import torch
from torch.autograd import Variable

from enums_and_constants import constants, SpecialTokens
from models import Transformer
from preload import preload_data_from_gdrive
from tokenizer import load_bpe_tokenizers
from word2vec import load_w2v_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    tool = language_tool_python.LanguageTool("en-US")
    preload_data_from_gdrive()
    src_tokenizer, trg_tokenizer = load_bpe_tokenizers(
        src_path=constants.SRC_TOKENIZER_PATH,
        trg_path=constants.TRG_TOKENIZER_PATH
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
        use_gate=False,
        eta=constants.ETA
    )
    state_dict = torch.load(os.path.join(constants.FULL_MODEL_CHKPT_PATH, "model.ckpt"))["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    telebot = telebot.TeleBot(constants.TELEGRAM_BOT_TOKEN)

    @telebot.message_handler(commands=["start"])
    def start(message):
        text = "Hi. This is simple RU-EN translation bot. You allow to use not very long sentences."
        telebot.send_message(message.chat.id, text=text)

    @telebot.message_handler(commands=["help"])
    def h(message):
        text = "Use command '/translate msg' to generate translation of msg."
        telebot.send_message(message.chat.id, text=text)

    @telebot.message_handler(commands=['translate'], content_types=['text'])
    def translate(message):
        sst = SpecialTokens.START_OF_SEQ.value['token']
        est = SpecialTokens.END_OF_SEQ.value['token']
        text = message.text.replace("/translate ", "")
        text = text.lower()
        for c in text:
            if c.isalpha() and c not in "абвгдежзийклмнопрстуфхцчшщъыьэюя":
                telebot.send_message(message.chat.id, text="Msg must contains cyrillic symbols.")
                return
        if text[-1].isalnum():
            text += "."
        text = sst + text + est
        with torch.no_grad():
            src = src_tokenizer.encode_batch([text])
            src = [x.ids for x in src]
            if len(src[0]) > constants.MAX_SEQ_LEN:
                telebot.send_message(message.chat.id, text="Too long message, try less words.")
            else:
                src = torch.LongTensor(src).to(device)
                src_mask = Transformer.src_mask(src)
                e_outputs, _ = model.encoder(src, src_mask)
                spi = SpecialTokens.PADDING.value['idx']
                ssi = SpecialTokens.START_OF_SEQ.value['idx']
                esi = SpecialTokens.END_OF_SEQ.value['idx']
                outs = torch.zeros(src.size(0), constants.MAX_SEQ_LEN).type_as(src.data).fill_(spi)
                outs[:, 0] = ssi
                for i in range(1, constants.MAX_SEQ_LEN):
                    trg_mask = np.triu(np.ones((1, i, i)), k=1).astype('uint8')
                    trg_mask = Variable(torch.from_numpy(trg_mask) == 0).to(device)
                    preds = model.out(model.decoder(outs[:, :i], e_outputs, src_mask, trg_mask)[0])
                    idx = preds[:, -1, :].argmax(dim=1)
                    outs[:, i] = idx
                    if outs[0][i] == esi:
                        break
                outs = outs.tolist()
                outs = trg_tokenizer.decode_batch(outs)
                text = tool.correct(outs[0])
                telebot.send_message(message.chat.id, text=text)

    telebot.polling(none_stop=True)
