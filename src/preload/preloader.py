import os.path

import gdown

from enums_and_constants import constants


def preload_data_from_gdrive() -> None:
    """ Preload transformer data from Google Drive.

    Returns
    -------
        None.
    """
    dataset_exist = os.path.exists(path=constants.DATASET_PATH)

    src_tokenizer_exist = os.path.exists(path=constants.SRC_TOKENIZER_PATH)
    trg_tokenizer_exist = os.path.exists(path=constants.TRG_TOKENIZER_PATH)

    src_w2v_exist = os.path.exists(path=constants.SRC_W2V_PATH)
    trg_w2v_exist = os.path.exists(path=constants.TRG_W2V_PATH)

    full_model_chkpt_exist = os.path.exists(path=constants.FULL_MODEL_CHKPT_PATH)
    full_model_logs_exist = os.path.exists(path=constants.FULL_MODEL_TRAIN_LOGS)

    prune_model_chkpt_exist = os.path.exists(path=constants.PRUNE_MODEL_CHKPT_PATH)
    prune_model_logs_exist = os.path.exists(path=constants.PRUNE_MODEL_TRAIN_LOGS)

    if dataset_exist and \
            src_tokenizer_exist and \
            trg_tokenizer_exist and \
            src_w2v_exist and \
            trg_w2v_exist and \
            full_model_chkpt_exist and \
            full_model_logs_exist and \
            prune_model_chkpt_exist and \
            prune_model_logs_exist:
        print("All data exists")
        return
    gdown.download_folder(url=constants.DATA_URL, output=constants.MAIN_DATA_PATH, quiet=False)
