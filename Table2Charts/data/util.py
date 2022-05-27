# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import pickle
from os import path

from .config import DataConfig


def load_json(file_path: str, encoding: str):
    with open(file_path, "r", encoding=encoding) as f:
        return json.load(f)


def load_pickle(file_path: str):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def get_embeddings(uID: str, config: DataConfig):
    if config.embed_in_json:
        return load_json(config.embedding_path(uID), config.encoding)
    else:
        return load_pickle(config.embedding_path(uID))


def load_mutual_information(mutual_info_path, config: DataConfig):
    mutual_information_dict = {}
    # path = os.path.join(corpus_path, cUID, "MI.pickle")
    if path.exists(mutual_info_path):
        mutual_information_dict = load_json(mutual_info_path, config.encoding)
    return mutual_information_dict
