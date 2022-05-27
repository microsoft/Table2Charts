# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from bisect import bisect_left
from collections import namedtuple
from copy import copy
from numpy import ndarray
from typing import List, Dict, Optional, Set

from .config import DataConfig
from .special_tokens import SpecialTokens
from .template import Template
from .token import Token, TokenType, Segment, AggFunc, FieldRole, GroupingOp, \
    IsPercent, IsCurrency, HasYear, HasMonth, HasDay

Result = namedtuple("Result", ["score", "state"])


def append_padding(seq: List, pad, final_length: int):
    return seq + [pad] * (final_length - len(seq))


class Sequence:
    """A list of token, segment pair. DO NOT use this for representing states! (Use State instead.)"""
    __slots__ = 'tokens', 'segments', 'hash_value', 'current'

    def __init__(self, tokens: List[Token], segments: List[Segment]):
        if len(tokens) != len(segments):
            raise ValueError("Lengths of tokens and segments not matching!")
        self.tokens = tokens
        self.segments = segments
        self.hash_value = self._calc_hash_(tokens)

    @staticmethod
    def _calc_hash_(tokens: List[Token]):
        value = 487
        for t in tokens:
            value = value * 31 + hash(t)
        return value

    def _update_hash_(self, t: Token):
        self.hash_value = self.hash_value * 31 + hash(t)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Sequence):
            if len(self.tokens) == len(o.tokens):
                for i in range(len(o.tokens)):
                    if self.tokens[i] != o.tokens[i]:
                        return False
                return True
        return False

    def __hash__(self) -> int:
        return self.hash_value

    def __repr__(self):
        return "(" + " ".join(t.__repr__() for t in self.tokens) + ")"

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item) -> Token:
        return self.tokens[item]

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self) -> Token:
        if self.current >= len(self.tokens):
            raise StopIteration
        else:
            self.current += 1
            return self[self.current - 1]

    def __copy__(self):
        return Sequence(self.tokens[:], self.segments[:])

    def num_fields(self):
        return sum(token.type is TokenType.FIELD for token in self.tokens)

    def unique_field_tokens(self):
        unique_field_tokens = list(set(filter(lambda t: t.type == TokenType.FIELD, self.tokens)))
        unique_field_tokens.sort()
        return unique_field_tokens

    @staticmethod
    def _compose_categories_(token: Token, config: DataConfig):
        categories = []
        # The adding order should be the same as config.cat_num order
        if config.use_field_type:
            categories.append(0 if token.field_type is None else int(token.field_type))
        if config.use_field_role:
            categories.append(FieldRole.to_int(token.field_role))
        if config.use_binary_tags:
            tags = tuple([None] * 5) if token.tags is None else token.tags
            categories.append(IsPercent.to_int(tags[0]))
            categories.append(IsCurrency.to_int(tags[1]))
            categories.append(HasYear.to_int(tags[2]))
            categories.append(HasMonth.to_int(tags[3]))
            categories.append(HasDay.to_int(tags[4]))
        if config.consider_grouping_operations:
            categories.append(GroupingOp.to_int(token.grp_op))
        if config.top_freq_func > 0:
            categories.append(AggFunc.to_int(token.agg_func))
        return categories

    def to_dict(self, final_len: int, field_permutation: Optional[ndarray],
                field_indices: bool, fixed_order: bool, config: DataConfig):
        # TODO: memory efficient way to construct tensors
        if field_permutation is None:
            tokens = self.tokens
            indices = [-1 if token.field_index is None else token.field_index for token in tokens]
        else:  # field permutation should be applied
            indices = [-1] * len(self.tokens)
            if fixed_order:  # The field token order in state should be fixed
                tokens = self.tokens
                reverse = [0] * len(field_permutation)
                for idx, origin in enumerate(field_permutation):
                    reverse[origin] = idx
                for i in range(len(tokens)):
                    if tokens[i].field_index is not None:
                        indices[i] = reverse[tokens[i].field_index]
            else:  # The field token order in action space should be changed
                tokens = copy(self.tokens)
                for idx, origin in enumerate(field_permutation):
                    tokens[idx] = self.tokens[origin]
                    indices[idx] = idx

        result = {
            "token_types": torch.tensor(append_padding([
                token.type.value if config.unified_ana_token else
                (token.ana_type.value + TokenType.ANA.value if token.type is TokenType.ANA else token.type.value)
                for token in tokens], 0, final_len), dtype=torch.long),  # Count AnaType variations as Token Types
            "segments": torch.tensor(append_padding([segment.value for segment in self.segments], 0, final_len),
                                     dtype=torch.long),
            "categories": torch.tensor(append_padding(
                [self._compose_categories_(token, config)
                 for token in tokens],
                config.empty_cat, final_len), dtype=torch.long),
            "semantic_embeds": 0 if len(config.empty_embed) == 0 else torch.tensor(append_padding(
                [config.empty_embed if token.semantic_embed is None else token.semantic_embed
                 for token in tokens],
                config.empty_embed, final_len), dtype=torch.float),
            "data_characters": 0 if len(config.empty_data) == 0 else torch.tensor(append_padding(
                [config.empty_data if token.data_features is None else token.data_features
                 for token in tokens],
                config.empty_data, final_len), dtype=torch.float),
            "mask": torch.tensor([1] * len(tokens) + [0] * (final_len - len(tokens)), dtype=torch.uint8)
        }
        if field_indices:
            result["field_indices"] = torch.tensor(append_padding(indices, -1, final_len), dtype=torch.long)
        return result


class State(Sequence):
    """Used for representation of state. Keep the logic of analysis language format."""
    __slots__ = 'template', 't_end'

    def __init__(self, template: Template):
        super().__init__([], [])

        self.template = template
        self.t_end = [0]

    @classmethod
    def init_state(cls, template: Template):
        state = State(template)
        state.append(template.first_token())
        return state

    @classmethod
    def fill_template(cls, template: Template, choices: Dict[Segment, List[Token]]):
        """
        Construct complete state (prefix) from specified segments.
        :return: constructed state.
        """
        state = State(template)
        for t_i in range(len(template)):
            token_type = template.types[t_i]
            segment = template.segments[t_i]
            if token_type is TokenType.ANA:
                state.append(template.first_token())
            elif token_type is TokenType.SEP:
                state.append(SpecialTokens.SEP_TOKEN)
            elif token_type is TokenType.FUNC:
                agg_func = choices[Segment.FUNC]  # segment == Segment.FUNC
                if len(agg_func) != 1 or not template.lower[t_i] <= len(agg_func) <= template.upper[t_i]:
                    raise ValueError("Input choices does not contain exactly 1 agg_func!")
                state.append(agg_func[0])
            elif token_type is TokenType.GRP:
                grp_op = choices[Segment.GRP]  # segment == Segment.GRP
                if len(grp_op) != 1 or not template.lower[t_i] <= len(grp_op) <= template.upper[t_i]:
                    raise ValueError("Input choices does not contain exactly 1 grp_op!")
                state.append(grp_op[0])
            elif token_type is TokenType.FIELD:  # Including VAL, DIM, CAT
                fields = choices[segment]
                if not template.lower[t_i] <= len(fields) <= template.upper[t_i]:
                    raise ValueError(f"{fields} does not meet the template bounds of "
                                     f"{template.lower[t_i]} and {template.upper[t_i]}.")
                if template.increasing[t_i]:
                    fields = sorted(fields, key=lambda x: x.field_index)
                    if template.contiguous[t_i]:
                        for i in range(1, len(fields)):
                            if fields[i].field_index - fields[i - 1].field_index != 1:
                                raise ValueError(f"Not contiguous fields for template[{t_i}]!")
                for field in fields:
                    if field.field_type in template.forbidden[t_i]:
                        raise ValueError(f"A forbidden field type {field.field_type} is used at template[{t_i}].")
                    state.append(field)
        if not state.is_complete():
            raise ValueError("State is not completed after filling!")
        return state

    def __copy__(self):
        new_state = State(self.template)
        new_state.tokens = self.tokens[:]
        new_state.segments = self.segments[:]
        new_state.hash_value = self.hash_value
        new_state.t_end = self.t_end[:]
        return new_state

    def prefix(self, length: int):
        prefix = State(self.template)
        prefix.tokens = self.tokens[:length]
        prefix.segments = self.segments[:length]
        prefix.hash_value = self._calc_hash_(prefix.tokens)
        prefix.t_end = self.t_end[:bisect_left(prefix.t_end, length) + 1]
        prefix.t_end[-1] = length
        return prefix

    def valid_actions(self, action_space: Sequence, top_freq_func: int,
                      max_rc: tuple = (1000, 1000), max_x: int = 1000,
                      selected_field_indices: Optional[Set[int]] = None) -> List[bool]:
        """
        Determine if each action in action space is valid under the (incomplete) state
        :param selected_field_indices: limit candidate fields to the given ones
        :param action_space: Sequence
        :param max_rc: max number of rows and columns
        :param max_x: max number of x fields
        :param top_freq_func: top k most frequent agg funcs
        :return: bool mask array
        """
        t_idx = len(self.t_end) - 1
        current_token_num = self.t_end[t_idx] - (0 if t_idx == 0 else self.t_end[t_idx - 1])
        # In charts there should be no duplications.
        unique_fields = {self.tokens[i] for i in range(len(self)) if self.tokens[i].type is TokenType.FIELD}
        if selected_field_indices is not None:  # Count the number of all used fields.
            field_cnt = len({self.tokens[i] for i in range(len(self)) if self.tokens[i].type is TokenType.FIELD})
        else:
            field_cnt = None
        valid_actions = []
        func_num = 0
        tp = self.template

        for action in action_space:
            # Find the index in template for the action
            steps = 0
            while self.template.types[t_idx + steps] != action.type:
                if steps == 0:
                    if not tp.lower[t_idx] <= current_token_num <= tp.upper[t_idx]:  # Current template not satisfied
                        steps = -1
                        break
                else:
                    if tp.lower[t_idx + steps] != 0:  # Cannot ignore the template token
                        steps = -1
                        break
                steps += 1
                current_token_num = 0

            if steps == -1:  # Not fit to the basic token type grammar
                valid_actions.append(False)
                continue
            else:
                t_idx += steps

            if current_token_num >= tp.upper[t_idx]:  # Exceed the upper limit of the template
                valid_actions.append(False)
                continue

            if selected_field_indices is not None and t_idx == len(tp) - 1 and \
                    field_cnt != len(selected_field_indices):
                # This token ends the template but not all selected fields are used.
                valid_actions.append(False)
                continue

            if action.type is TokenType.FIELD:
                if tp.increasing[t_idx] and current_token_num != 0:
                    if self.tokens[-1].field_index >= action.field_index:
                        valid_actions.append(False)  # Not follow increasing indices
                        continue
                    if tp.contiguous[t_idx] and self.tokens[-1].field_index != action.field_index - 1:
                        valid_actions.append(False)  # Not the adjacent field to the previous one
                        continue
                if action.field_type in tp.forbidden[t_idx]:
                    valid_actions.append(False)  # Forbidden field type
                    continue
                if tp.segments[t_idx] is Segment.ROW:
                    if current_token_num >= max_rc[0]:
                        valid_actions.append(False)  # Too many row fields
                        continue
                elif tp.segments[t_idx] is Segment.COL:
                    if current_token_num >= max_rc[1]:
                        valid_actions.append(False)  # Too many column fields
                        continue
                elif tp.segments[t_idx] is Segment.X:
                    if current_token_num >= max_x:
                        valid_actions.append(False)  # Too many x fields
                        continue
                if selected_field_indices is not None and action.field_index not in selected_field_indices:
                    valid_actions.append(False)  # Not a selected field
                    continue
                valid_actions.append(action not in unique_fields)  # No duplications in dimensions or categories
            elif action.type is TokenType.GRP:  # Only available for BarChart when consider_grouping_operations
                # Stack is only available for multi-value BarChart
                valid_actions.append(action.grp_op is GroupingOp.Cluster or self.t_end[1] - self.t_end[0] > 1)
            elif action.type is TokenType.FUNC:  # Only available for PivotTable, tokens[1] is the measure
                func_num += 1
                valid_actions.append(func_num <= top_freq_func and self.tokens[1].compatible_with(action.agg_func))
            elif action.type is TokenType.SEP:
                valid_actions.append(True)
            else:
                raise ValueError(f"Unexpected action {action}!")

        return valid_actions

    def append(self, token: Token):
        t_idx = len(self.t_end) - 1
        pos = self.t_end[t_idx]
        steps = self.template.move_steps(t_idx, 0 if t_idx == 0 else self.t_end[t_idx - 1], pos, token)

        self.tokens.append(token)
        self.segments.append(self.template.segments[t_idx + steps])
        self._update_hash_(token)
        self.t_end.extend([pos] * steps)
        self.t_end[-1] += 1

        return self

    def is_complete(self) -> bool:
        """
        Check if the token sequence is complete: represents a full analysis.
        :return: bool
        """
        return len(self.t_end) == len(self.template) and \
               self.template.lower[-1] <= self.t_end[-1] - self.t_end[-2] <= self.template.upper[-1]

    def dissect(self) -> Dict[Segment, List[Token]]:
        results = {}
        for segment, indices in self.template.seg_idx.items():
            if segment != Segment.OP:
                tokens = []
                for idx in indices:
                    start = 0 if idx == 0 else self.t_end[idx - 1]
                    end = self.t_end[idx]
                    for j in range(start, end):
                        tokens.append(self.tokens[j])
                if segment not in results:
                    results[segment] = tokens
                else:
                    results[segment].extend(tokens)
        return results

    def stage(self):
        """
        :return: the number of non-field tokens - 1
        """
        return sum(t.type is not TokenType.FIELD for t in self.tokens) - 1

    def is_stage_checkpoint(self):
        """
        :return: If current state is just at the end of a complete stage.
        """
        return self.tokens[-1].type is not TokenType.FIELD

    def to_json(self):
        convert = self.dissect()
        output = {TokenType.ANA.name: self.template.ana.name}
        for key, value in convert.items():
            if isinstance(value, list):
                output[key.name] = [repr(v) for v in value]
            else:
                output[key.name] = repr(value)
        return output
