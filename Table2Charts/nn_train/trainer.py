# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn import metrics
from timeit import default_timer as timer
from torch.cuda.streams import Stream
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    from apex import amp
except ImportError:
    pass

from util import to_device, scores_from_confusion

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class Trainer:
    def __init__(self, ddp: nn.Module, device, rank: int, use_apex: bool,
                 train_loader: DataLoader, test_loader: DataLoader, epoch_init_fn,
                 optimizer: Optimizer, criterion: nn.Module, summary_writer: SummaryWriter,
                 log_freq: int = 10):
        """
        :param ddp: the model for training, should already wrapped on the corresponding device
        :param train_loader: train dataset loader
        :param test_loader: test dataset loader
        :param log_freq: logging frequency of the batch iteration
        """
        self.model = ddp
        self.device = device
        self.rank = rank
        self.apex = use_apex

        self.train_data = train_loader
        self.test_data = test_loader
        self.epoch_fn = epoch_init_fn

        self.optim = optimizer
        self.criterion = criterion
        self.log_freq = log_freq
        self.summary_writer = summary_writer

    def train(self, epoch: int):
        return self.iteration(epoch, self.train_data)

    def test(self, epoch: int):
        return self.iteration(epoch, self.test_data, is_train=False)

    def iteration(self, epoch: int, data_loader: DataLoader, is_train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param is_train: boolean value of is train or test
        """
        str_code = f"train@{os.getpid()}" if is_train else f"test@{os.getpid()}"
        logger = logging.getLogger(f"EP{epoch}_{str_code}")

        self.model.train(is_train)
        self.epoch_fn(epoch)  # By default this is calling the set_epoch() of DistributedSampler.
        loss_sum = 0.0
        confusion_sum = np.zeros((2, 2), dtype=int)
        start = timer()

        total_batch = len(data_loader)
        with torch.set_grad_enabled(is_train):
            i = 0
            for data in self.pipeline(data_loader):
                loss, matrix = self.calculate(data, is_train)

                loss_sum += loss
                confusion_sum += matrix

                i += 1

                info = {
                    "epoch": epoch,
                    "iter": "[%d/%d]" % (i, total_batch),
                    "loss": loss,
                    "avg_loss": loss_sum / i,
                    "prf1": scores_from_confusion(matrix),
                    "all_prf1": scores_from_confusion(confusion_sum),
                    "all_acc": np.trace(confusion_sum) / np.sum(confusion_sum),
                }

                if i % self.log_freq == 0:
                    end = timer()

                    logger.info(
                        "%s loss=%f prf1=%s" % (info["iter"], info["loss"], info["prf1"]) +
                        " | all_acc=%f avg_loss=%f all_prf1=%s" % (info["all_acc"], info["avg_loss"], info["all_prf1"])
                        + " | elapsed=%.1fs" % (end - start))

                if self.rank == 0:
                    summary_type = "train" if is_train else "test/valid"
                    global_step = epoch * total_batch + i

                    self.summary_writer.add_scalar(f"{summary_type}/loss", info["loss"], global_step)
                    self.summary_writer.add_scalar(f"{summary_type}/avg_loss", info["avg_loss"], global_step)
                    self.summary_writer.add_scalar(f"{summary_type}/precision", info["prf1"][0], global_step)
                    self.summary_writer.add_scalar(f"{summary_type}/recall", info["prf1"][1], global_step)
                    self.summary_writer.add_scalar(f"{summary_type}/f1_score", info["prf1"][2], global_step)
                    self.summary_writer.add_scalar(f"{summary_type}/all_precision", info["all_prf1"][0], global_step)
                    self.summary_writer.add_scalar(f"{summary_type}/all_recall", info["all_prf1"][1], global_step)
                    self.summary_writer.add_scalar(f"{summary_type}/all_f1_score", info["all_prf1"][2], global_step)
                    self.summary_writer.add_scalar(f"{summary_type}/all_acc", info["all_acc"], global_step)
                    self.summary_writer.flush()

        end = timer()
        # Reduce evaluation metrics. Sync point!
        loss_sum_tensor = torch.tensor(loss_sum, device=self.device, dtype=torch.double)
        confusion_sum_tensor = torch.tensor(confusion_sum, device=self.device, dtype=torch.int64)
        dist.all_reduce(loss_sum_tensor)
        dist.all_reduce(confusion_sum_tensor)
        loss_sum = loss_sum_tensor.item()
        confusion_sum = confusion_sum_tensor.cpu().numpy()

        if self.rank == 0:
            avg_loss = loss_sum / total_batch
            precision, recall, f1 = scores_from_confusion(confusion_sum)

            logger.info("elapsed=%.1fs, avg_loss=%f " % (end - start, avg_loss) +
                        "(tn, fp, fn, tp)=%s " % confusion_sum.ravel() +
                        "precision=%f recall=%f f1=%f" % (precision, recall, f1))

            summary_type = ("train" if is_train else "test/valid") + "-summary"
            self.summary_writer.add_scalar(f"{summary_type}/elapsed_time", end - start, epoch)
            self.summary_writer.add_scalar(f"{summary_type}/avg_loss", avg_loss, epoch)
            self.summary_writer.add_scalar(f"{summary_type}/precision", precision, epoch)
            self.summary_writer.add_scalar(f"{summary_type}/recall", recall, epoch)
            self.summary_writer.add_scalar(f"{summary_type}/f1_score", f1, epoch)
            self.summary_writer.flush()

        return loss_sum, total_batch, confusion_sum

    def calculate(self, data, is_train: bool):
        with torch.set_grad_enabled(is_train):
            # 0. batch_data will be sent into the device(GPU or cpu)
            # data = to_device(data, self.device, non_blocking=True)

            # 1. forward passing
            output = self.model(data["state"], data["actions"])
            # print(output.size(), output)

            # 2. NLL(negative log likelihood) loss of binary classification result
            target = data["values"]
            loss = self.criterion(output.transpose(-1, -2), target)

            # 3. backward and optimization only in train
            if is_train:
                self.optim.zero_grad()
                if self.apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()  # Sync point!
                else:
                    loss.backward()  # Sync point!
                self.optim.step()

            # 4. evaluation metrics
            y_pred = output.detach().argmax(dim=-1).cpu().numpy().ravel()
            y_true = target.cpu().numpy().ravel()
            valid_b = (y_true != -1)
            y_pred = y_pred[valid_b]
            y_true = y_true[valid_b]
            matrix = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])

            loss_val = loss.item()
            del loss, output, data
            return loss_val, matrix

    def pipeline(self, loader: DataLoader):
        """
        Software pipeline to help load next batch while the previous batch is being processed.
        See https://dev.azure.com/msresearch/GCR/_wiki/wikis/GCR.wiki/3390/Pre-fetching-(Software-Pipelining)
        """
        stream = Stream(self.device)
        first = True
        data = None
        for next_data in loader:
            with torch.cuda.stream(stream):
                next_data = to_device(next_data, self.device, non_blocking=True)
            if not first:
                yield data
            else:
                first = False
            torch.cuda.current_stream().wait_stream(stream)
            data = next_data
        yield data
