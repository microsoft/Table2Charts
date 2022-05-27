# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import numpy as np
import os
import torch
import torch.distributed as dist
from enum import IntEnum
from helper import save_ddp_checkpoint
from sklearn import metrics
from time import perf_counter, process_time
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
from typing import List, Callable

try:
    from apex import amp
except ImportError:
    pass

from data import DataConfig, SpecialTokens, QValue, State, Sequence, get_template
from search.agent import BeamDrillDownAgent, ParallelAgents, SearchConfig
from util import to_device, scores_from_confusion, time_str
from .config import StudentConfig
from .noise import OUNoise
from .replay_memory import ReplayMemory

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class BatchMode(IntEnum):
    Estimate = 0,
    Train = 1,
    Test = 2


class Student:
    def __init__(self, config: StudentConfig, data_config: DataConfig, search_config: SearchConfig,
                 ddp: Module, use_apex: bool, device, local_rank, summary_writer: SummaryWriter):
        self.config = config
        self.data_config = data_config
        self.search_config = search_config
        self.special_tokens = SpecialTokens(data_config)

        self.model = ddp
        self.optimizer = config.optimizer
        self.criterion = config.criterion
        self.apex = use_apex

        self.local_rank = local_rank
        self.device = device
        self.agents = ParallelAgents()
        self.memory = ReplayMemory(config.memory_size)

        self.noise_scale = 1
        self.noise = OUNoise(data_config.max_field_num + data_config.num_cmd_tokens())

        self.current_epoch = -1
        self.is_testing = False
        self.start_perf_time = 0
        self.start_process_time = 0
        self.batch_cnt = 0
        self.loss_sum = 0.0
        self.confusion_sum = np.zeros((2, 2), dtype=int)

        self.logger = logging.getLogger(f"Student {config.log_tag}")
        self.log_freq = config.log_freq
        self.summary_writer = summary_writer
        self.global_train_step = 0
        self.global_test_step = 0

        # Pass any template is OK here
        fake_state = State.init_state(get_template(data_config.search_types[0],
                                                   data_config.allow_multiple_values,
                                                   data_config.consider_grouping_operations,
                                                   data_config.limit_search_group))
        fake_actions = Sequence([], [])
        samples = [QValue(fake_state, fake_actions, [], [])]
        self.fake_data = self._prepare_data_(samples, only_estimate=True)

    def add_table(self, tUID: str):
        # TODO: add argument to choose from search agents, define a create_agent() func
        self.agents.add(BeamDrillDownAgent(tUID, self.data_config, self.special_tokens, self.search_config))

    def n_tables(self):
        return self.agents.remaining() + self.agents.finished()

    def reset(self, epoch: int, is_testing: bool):
        self.start_perf_time = perf_counter()
        self.start_process_time = process_time()
        self.batch_cnt = 0
        self.loss_sum = 0.0
        self.confusion_sum = np.zeros((2, 2), dtype=int)

        self.current_epoch = epoch
        self.is_testing = is_testing

        self.agents.shutdown()
        self.agents = ParallelAgents()

        self.noise.reset()
        if self.noise_scale == 1:
            self.noise_scale = self.config.scale_start
        else:
            self.noise_scale = max(self.config.scale_end, self.noise_scale * self.config.scale_decay)

    def metrics(self):
        return self.loss_sum, self.batch_cnt, self.confusion_sum, self.start_perf_time

    def save_checkpoint(self, dir_path: str):
        return save_ddp_checkpoint(dir_path, self.current_epoch, self.model, self.optimizer)

    def _prepare_data_(self, samples: List[QValue], only_estimate: bool = False):
        d_config = self.data_config
        data = QValue.collate(samples, d_config, not self.is_testing and d_config.field_permutation)
        if only_estimate:
            data.pop("values")  # Not useful in pure estimation
        data = to_device(data, self.device)
        return data

    def _feed_batch_nn_(self, mode: BatchMode, samples: List[QValue], noise_fn: Callable = None) -> np.ndarray:
        self.model.train(mode is BatchMode.Train)

        if len(samples) == 0:  # We still feed a fake batch to keep DDP in sync!
            with torch.no_grad():
                self.model(self.fake_data["state"], self.fake_data["actions"])  # Sync point!
            return np.empty([0])

        with torch.set_grad_enabled(mode is BatchMode.Train):
            if mode is BatchMode.Estimate:  # Return action value estimations
                data = self._prepare_data_(samples, only_estimate=True)
                # No evaluation metrics during estimation
                output = self.model(data["state"], data["actions"])  # Sync point!
                estimates = output.detach()[:, :, 1].exp().cpu().numpy()
                # Add noise for exploration
                m, n = estimates.shape
                noises = np.array([noise_fn(n) for _ in range(m)])
                return np.clip(estimates + noises, 0, 1)
            else:  # Train on experiences or Test on validation/test data
                data = self._prepare_data_(samples)
                output = self.model(data["state"], data["actions"])  # Sync point!
                target = data["values"]
                loss = self.criterion(output.transpose(-1, -2), target)

                if mode is BatchMode.Train:
                    self.optimizer.zero_grad()
                    if self.apex:
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()  # Sync point!
                    else:
                        loss.backward()  # Sync point!
                    self.optimizer.step()

                self.batch_cnt += 1
                self.loss_sum += loss.item()
                y_pred = output.detach().argmax(dim=-1).cpu().numpy().ravel()
                y_true = target.cpu().numpy().ravel()
                valid_b = (y_true != -1)
                y_pred = y_pred[valid_b]
                y_true = y_true[valid_b]
                matrix = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
                self.confusion_sum += matrix

                ap, ar, af1 = scores_from_confusion(self.confusion_sum)
                tfnp = self.confusion_sum.ravel()
                if self.batch_cnt % self.log_freq == 0:
                    phase_tag = "test/valid" if self.is_testing else "train"
                    self.logger.info(f"EP-{self.current_epoch} {phase_tag} B{self.batch_cnt}: loss={loss.item()}" +
                                     "| (tn, fp, fn, tp)=%s all_prf1=(%f, %f, %f) avg_loss=%f all_acc=%f" % (
                                         tfnp, ap, ar, af1,
                                         self.loss_sum / self.batch_cnt,
                                         np.trace(self.confusion_sum) / np.sum(self.confusion_sum)) +
                                     "| process=%.1fs elapsed=%.1fs" % (
                                         process_time() - self.start_process_time,
                                         perf_counter() - self.start_perf_time))

                if self.local_rank == 0:
                    if self.is_testing:
                        self.global_test_step += 1
                        global_step = self.global_test_step
                    else:
                        self.global_train_step += 1
                        global_step = self.global_train_step
                    phase_tag = 'test/valid' if self.is_testing else 'train'
                    self.summary_writer.add_scalar("{0}/loss".format(phase_tag), loss.item(), global_step)
                    self.summary_writer.add_scalar("{0}/avg_loss".format(phase_tag), self.loss_sum / self.batch_cnt,
                                                   global_step)
                    self.summary_writer.add_scalar("{0}/all_acc".format(phase_tag),
                                                   np.trace(self.confusion_sum) / np.sum(self.confusion_sum),
                                                   global_step)
                    self.summary_writer.add_scalar("{0}/precision".format(phase_tag), ap, global_step)
                    self.summary_writer.add_scalar("{0}/recall".format(phase_tag), ar, global_step)
                    self.summary_writer.add_scalar("{0}/f1_score".format(phase_tag), af1, global_step)
                    self.summary_writer.flush()

                del loss

                if mode is BatchMode.Test:
                    return output.detach()[:, :, 1].exp().cpu().numpy()

    @staticmethod
    def _feed_batch_random_(samples: List[QValue]) -> List:  # Only call this for BatchMode.Estimate!
        return [np.clip(sample.values + np.random.rand(len(sample.values)), 0, 1) for sample in samples]

    def _act_step_(self, samples: List[QValue], noise_fn: Callable):
        useful_samples = [sample for sample in samples if sample.has_valid_action]
        for sample in useful_samples:
            self.memory.push(sample)
        if not self.is_testing and self.config.random_train:
            # Take random action (except for targets) to generate random samples.
            # Notice: this will lead to recall during training always 100%!
            return self._feed_batch_random_(samples)
        useful_results = self._feed_batch_nn_(BatchMode.Test if self.is_testing else BatchMode.Estimate,
                                              useful_samples, noise_fn)
        results = []
        idx = 0
        for sample in samples:
            if sample.has_valid_action:
                results.append(useful_results[idx])
                idx += 1
            else:  # any value is ok since it will not be taken
                results.append(sample.values)
        return results

    def _noise_fn_(self, length: int):
        # Add noise to estimate results during exploration phase of RL
        return self.noise_scale * self.noise.sample()[:length]

    def act_step(self):
        # Each search agent will return a beam of states for estimating
        futures = self.agents.step([lambda samples: self._act_step_(samples, self._noise_fn_)])  # Work on only 1 GPU

        # Return the estimating results to agents for updating
        finished_info = self.agents.update([(lambda: future.result()) for future in futures])
        return finished_info

    def sample_learn(self, rounds: int, batch_size: int):
        for _ in range(rounds):
            samples = self.memory.sample(batch_size)
            self._feed_batch_nn_(BatchMode.Train, samples)

    def dist_summary(self):  # Sync point!
        end_perf_time = perf_counter()
        loss_sum_tensor = torch.tensor(self.loss_sum, device=self.device, dtype=torch.double)
        confusion_sum_tensor = torch.tensor(self.confusion_sum, device=self.device, dtype=torch.int64)

        merged_info = self.agents.summary(divide_total=False)
        log_dir_path = os.path.join(self.config.log_dir, "test-valid" if self.is_testing else "train")
        os.makedirs(log_dir_path, exist_ok=True)
        log_file_path = os.path.join(log_dir_path,
                                     f"[summary-{self.current_epoch:02d}]{time_str()}.rank-{self.local_rank}.log")
        with open(log_file_path, "w") as log_file:
            json.dump(merged_info, log_file, sort_keys=True, indent=4)

        recall01ordered = merged_info['evaluation']['stages']["complete"]['recall']['@01']
        recall03ordered = merged_info['evaluation']['stages']["complete"]['recall']['@03']
        recall05ordered = merged_info['evaluation']['stages']["complete"]['recall']['@05']
        recall10ordered = merged_info['evaluation']['stages']["complete"]['recall']['@10']

        info_tensor = torch.tensor([
            merged_info['expanded_states'], merged_info['reached_states'],
            merged_info["cut_states"], merged_info["dropped_states"], merged_info["complete_states"],
            merged_info['perf_time'], merged_info['process_time'],
            recall01ordered, recall03ordered, recall05ordered, recall10ordered
        ], device=self.device, dtype=torch.double)

        success = merged_info['t_cnt']
        final_stage_cnt = merged_info['evaluation']['stages']["complete"]['t_cnt']
        cnt_tensor = torch.tensor([self.batch_cnt, success, final_stage_cnt], device=self.device, dtype=torch.int64)

        # Reduce evaluation metrics. Sync point!
        dist.all_reduce(loss_sum_tensor)
        dist.all_reduce(cnt_tensor)
        dist.all_reduce(confusion_sum_tensor)
        dist.all_reduce(info_tensor)

        if self.local_rank == 0:
            avg_loss = loss_sum_tensor.item() / cnt_tensor[0].item() if cnt_tensor[0].item() > 0 else 0
            confusion_sum = confusion_sum_tensor.cpu().numpy()
            precision, recall, f1 = scores_from_confusion(confusion_sum)
            info_tensor = (info_tensor[:, 0] / info_tensor[:, 1]).cpu().numpy()

            tfnp = confusion_sum.ravel()
            self.logger.info(f"EP-{self.current_epoch} {'test/valid' if self.is_testing else 'train'} SUMMARY: " +
                             "elapsed=%.1fs | avg_loss=%f " % (end_perf_time - self.start_perf_time, avg_loss) +
                             "(tn, fp, fn, tp)=%s " % tfnp +
                             "precision=%f recall=%f f1=%f | " % (precision, recall, f1) +
                             f"total_cnt={cnt_tensor[0].item()}"
                             f"success_cnt={cnt_tensor[1].item()} "
                             f"#states(expanded, reached, cut, dropped, complete)="
                             f"({info_tensor[0]:.2f}, {info_tensor[1]:.2f}, "
                             f"{info_tensor[2]:.2f}, {info_tensor[3]:.2f}, {info_tensor[4]:.2f}) " +
                             f"t(perf, process)=({info_tensor[5]:.2f}s, {info_tensor[6]:.2f}s) " +
                             f"final_stage_cnt={cnt_tensor[2].item()} R@1={info_tensor[7]} " +
                             f"R@3={info_tensor[8]} R@5={info_tensor[9]} R@10={info_tensor[10]}")

            phase_tag = ('test/valid' if self.is_testing else 'train') + "-summary"
            global_step = self.current_epoch
            self.summary_writer.add_scalar("{0}/tn".format(phase_tag), tfnp[0], global_step)
            self.summary_writer.add_scalar("{0}/fp".format(phase_tag), tfnp[1], global_step)
            self.summary_writer.add_scalar("{0}/fn".format(phase_tag), tfnp[2], global_step)
            self.summary_writer.add_scalar("{0}/tp".format(phase_tag), tfnp[3], global_step)
            self.summary_writer.add_scalar("{0}/precision".format(phase_tag), precision, global_step)
            self.summary_writer.add_scalar("{0}/recall".format(phase_tag), recall, global_step)
            self.summary_writer.add_scalar("{0}/f1".format(phase_tag), f1, global_step)
            self.summary_writer.add_scalar("{0}/R@1_ordered".format(phase_tag), info_tensor[7], global_step)
            self.summary_writer.add_scalar("{0}/R@3_ordered".format(phase_tag), info_tensor[8], global_step)
            self.summary_writer.add_scalar("{0}/R@5_ordered".format(phase_tag), info_tensor[9], global_step)
            self.summary_writer.add_scalar("{0}/R@10_ordered".format(phase_tag), info_tensor[10], global_step)
            self.summary_writer.flush()
