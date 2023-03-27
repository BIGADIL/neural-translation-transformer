import os.path

MAIN_DATA_PATH = os.path.join("..", "..", "data")

DATASET_PATH = os.path.join(MAIN_DATA_PATH, "rus.txt")

SRC_TOKENIZER_PATH = os.path.join(MAIN_DATA_PATH, "tokenizers", "src_tokenizer.json")
TRG_TOKENIZER_PATH = os.path.join(MAIN_DATA_PATH, "tokenizers", "trg_tokenizer.json")

SRC_W2V_PATH = os.path.join(MAIN_DATA_PATH, "w2v", "src_w2v.bin")
TRG_W2V_PATH = os.path.join(MAIN_DATA_PATH, "w2v", "trg_w2v.bin")

FULL_MODEL_CHKPT_PATH = os.path.join(MAIN_DATA_PATH, "full_model", "chkpt")
FULL_MODEL_TRAIN_LOGS = os.path.join(MAIN_DATA_PATH, "full_model", "train_logs")

PRUNE_MODEL_CHKPT_PATH = os.path.join(MAIN_DATA_PATH, "prune_model", "chkpt")
PRUNE_MODEL_TRAIN_LOGS = os.path.join(MAIN_DATA_PATH, "prune_model", "train_logs")

DATA_URL = "https://drive.google.com/drive/folders/1zVsotEzUDgA-j1SHhBPVeehO_AiA5Lq7?usp=sharing"

TELEGRAM_BOT_TOKEN = "your token"

MODEL_DIM = 512
NUM_ENC_DEC_LAYERS = 4
HEADS = 8
MAX_SEQ_LEN = 20
ETA = 0.005
