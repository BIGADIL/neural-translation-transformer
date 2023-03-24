import gdown

from enums_and_constants import constants


def preload_data_from_gdrive():
    gdown.download_folder(url=constants.DATA_URL, output=constants.MAIN_DATA_PATH, quiet=False)
