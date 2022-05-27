# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch.nn import Module
from torch.optim.optimizer import Optimizer


class StudentConfig:
    def __init__(self, optimizer: Optimizer, loss: Module,
                 memory_size: int, min_memory: int, random_train: bool,
                 log_tag: str, log_freq: int, log_dir: str):
        self.optimizer = optimizer
        self.criterion = loss

        self.scale_start = 0.9
        self.scale_end = 0.001
        self.scale_decay = 0.8

        self.memory_size = memory_size
        self.min_memory = min_memory

        self.random_train = random_train

        self.log_tag = log_tag
        self.log_freq = log_freq
        self.log_dir = log_dir
