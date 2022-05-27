# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import os
import torch
from collections import defaultdict
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data._utils.collate import default_collate
from torch.utils.data._utils.pin_memory import pin_memory
from tqdm import tqdm
from typing import List, Optional, Tuple, Set, Iterable

from .config import DataConfig
from .dataset import DataTable, Chart
from .sequence import Sequence, State
from .special_tokens import SpecialTokens
from .token import AnaType, Token


class ChartUserActions(Dataset):
    __slots__ = "chart", "seq_len"

    def __init__(self, cUID: str, ana_type: AnaType, idx_to_field: dict,
                 config: DataConfig, search_sampling: bool):
        # TODO: remove this try-catch. (Work item #63)
        try:
            self.chart = Chart(cUID, ana_type, idx_to_field, config, search_sampling)
            self.seq_len = self.chart.seq_len()
        except:
            self.seq_len = 0

    def __len__(self):
        return len(self.chart.complete_states()) * (self.seq_len - 1)

    def __getitem__(self, index) -> Tuple[State, Token]:
        state_idx = index // (self.seq_len - 1)
        state_len = index % (self.seq_len - 1) + 1
        state = self.chart.states[state_idx].prefix(state_len)
        action = self.chart.states[state_idx][state_len]
        return state, action


class QValue:
    __slots__ = 'state', 'actions', 'valid_mask', 'has_valid_action', 'values', 'n_fields'

    def __init__(self, state: State, action_space: Sequence, valid_actions: Iterable[bool], action_values: Iterable):
        """
        :param state: a token sequence describing the state
        :param action_space: the whole action space (context), including invalid ones for the state
        :param valid_actions: a bool list indicating if an action in the action_space is valid
        :param action_values: 1 if the action leads to the final reward
        """
        self.state = state
        self.actions = action_space  # The field tokens should always be the first n_fields ones.
        self.valid_mask = np.array(valid_actions)
        self.has_valid_action = any(valid_actions)
        self.values = np.array(action_values)
        self.n_fields = action_space.num_fields()

    def __hash__(self) -> int:
        return hash(self.state)

    def __len__(self):
        return len(self.state)

    def __copy__(self):
        return QValue(self.state, self.actions, self.valid_mask, self.values)

    def to_dict(self, state_len: int, action_len: int, field_permutation: bool, config: DataConfig):
        # Necessary preparations for "values" tensor
        values = self.values.copy()
        values[np.logical_not(self.valid_mask)] = -1  # Only 0, 1 for valid actions
        if field_permutation:  # randomly permute field order in a table
            permutation = np.random.permutation(self.n_fields)
            values[:self.n_fields] = values[permutation]  # we only need to focus on the field tokens
        else:
            permutation = None

        return {
            "state": self.state.to_dict(state_len, permutation, config.need_field_indices, False, config),
            "actions": self.actions.to_dict(action_len, permutation, False, True, config),
            "values": torch.tensor(np.pad(values, (0, action_len - len(values)), mode='constant', constant_values=-1),
                                   dtype=torch.long)
        }

    @staticmethod
    def collate(batch, config: DataConfig, field_permutation: bool, pin: bool = False):
        state_len = max(map(lambda x: len(x.state), batch))
        action_len = max(map(lambda x: len(x.actions), batch))

        batch = default_collate([x.to_dict(state_len, action_len, field_permutation, config) for x in batch])
        return pin_memory(batch) if pin else batch


def determine_action_values(action_space: Sequence, positive_actions: Optional[Set[Token]]):
    if positive_actions:
        return [1 if action in positive_actions else 0 for action in action_space]
    else:
        return [0] * len(action_space)


class TableQValues(Dataset):
    def __init__(self, tUID: str, special_tokens: SpecialTokens, config: DataConfig, search_sampling: bool = False):
        self.table = DataTable(tUID, special_tokens, config)

        self.complete_states = set()
        self.state_actions = defaultdict(set)

        self.valid_p = 0
        self.valid_c = 0

        for cUID, cType in zip(self.table.cUIDs, self.table.cTypes):
            if cType not in config.input_types:
                continue
            chart = ChartUserActions(cUID, cType, self.table.idx2field, config, search_sampling)
            if chart.seq_len == 0:
                continue

            for state, action in chart:
                self.state_actions[state].add(action)
            chart_complete_states = chart.chart.complete_states()
            if search_sampling:
                self.complete_states.update(chart_complete_states)
            if len(chart_complete_states) > 0:
                self.valid_c += 1

        self.samples = list(self.state_actions.items())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index) -> QValue:
        state, positive_actions = self.samples[index]
        valid_actions = state.valid_actions(self.table.action_space, top_freq_func=self.table.config.top_freq_func)
        action_values = determine_action_values(self.table.action_space, positive_actions)

        return QValue(state, self.table.action_space, valid_actions, action_values)

    def get_state_actions(self):
        return self.state_actions

    def get_positive_prefixes(self):
        return self.complete_states | self.get_state_actions().keys()


class QValueDataset(ConcatDataset):
    def __init__(self, tUIDs: List[str], config: DataConfig, is_train: bool,
                 search_sampling: bool = False, tqdm_desc: str = None):
        """
        Create a concatenated dataset that contains TableQValues of all given tUIDs
        :param tUIDs: The tables to contain in this dataset
        :param config: DataConfig
        :param is_train: If this dataset is used for training, only useful when config.field_permutation
        :param search_sampling: For search sampling, generate all permutations and
        cache complete states in each TableQValues
        :param tqdm_desc: Print TQDM description for debugging
        """
        self.tUIDs = tUIDs
        self.config = config
        self.is_train = is_train

        special_tokens = SpecialTokens(config)

        tUIDs = tqdm(self.tUIDs, desc=tqdm_desc) if tqdm_desc is not None else self.tUIDs
        tables = []
        self.table_tqv = {}
        for tUID in tUIDs:
            tableQValues = TableQValues(tUID, special_tokens, config, search_sampling)
            # print(f"tUID={tUID}, len(tableQValues)={len(tableQValues)}")
            if search_sampling:
                self.table_tqv[tUID] = tableQValues
            if len(tableQValues) == 0:
                continue
            tables.append(tableQValues)

        # Randomly choosing [num_train_analysis] samples for a type.
        # For example, if the smallest number of one single type is 100k, and there are 500k LineCharts,
        # we'll randomly take 100k LineCharts to train.
        if is_train and config.num_train_analysis is not None:
            # TODO: check the usage of this part and if they are necessary. (from PR 29 comment)
            # Need to be updated with current code logic
            part_tables = []
            num_sampled_analysis = 0
            permutation = []
            for i in range(10):
                permutation.extend(range(i, len(tables), 10))
            for tableIndex in permutation:
                part_tables.append(tables[tableIndex])
                num_sampled_analysis += (tables[tableIndex].valid_p + tables[tableIndex].valid_c)
                if num_sampled_analysis >= config.num_train_analysis:
                    break
            tables = part_tables

        super().__init__(tables)

    def get_table_tqv(self, tUID: str):
        return self.table_tqv[tUID]

    def collate(self, batch, pin_memory: bool = False):
        return QValue.collate(batch, self.config, self.is_train and self.config.field_permutation, pin_memory)
