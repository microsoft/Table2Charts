from typing import List, Union, Tuple

import numpy as np

from .special_tokens import SpecialTokens
from .token import Token, TokenType, AnaType, FieldType, Segment


class Template:
    def __init__(self, ana_type: AnaType, token_types: List[TokenType], segments: List[Segment],
                 lower_nums: List[int], upper_nums: List[Union[int, float]],
                 contiguous_fields: List[bool], increasing_indices: List[bool],
                 forbidden_types: List[Tuple]):
        self.ana = ana_type
        if len(token_types) != len(segments) or len(segments) != len(lower_nums) or len(lower_nums) != len(upper_nums):
            raise IndexError("Lengths of grammar definition arrays are not the same!")
        self.types = token_types
        self.segments = segments
        self.lower = lower_nums
        self.upper = upper_nums
        self.contiguous = contiguous_fields
        self.increasing = increasing_indices
        self.forbidden = forbidden_types

        self.seg_idx = {}
        for i, s in enumerate(segments):
            if s not in self.seg_idx:
                self.seg_idx[s] = [i]
            else:
                self.seg_idx[s].append(i)

    def __len__(self):
        return len(self.types)

    def first_token(self):
        return SpecialTokens.get_ana_token(self.ana)

    def segment_idx(self, segment: Segment):
        return self.seg_idx[segment]

    def move_steps(self, idx: int, start: int, end: int, token: Token) -> int:
        # TODO: also check contiguous and increasing in this function.
        # TODO: also check forbidden types in this function
        if token.type == self.types[idx]:
            if end - start < self.upper[idx]:
                return 0
        if end - start < self.lower[idx]:
            raise ValueError(f"{token} do not follow {self.ana.name} template lower bound at index {idx}.")
        steps = 1
        while token.type != self.types[idx + steps]:
            if self.lower[idx + steps] > 0:
                raise ValueError(f"{token} do not follow {self.ana.name} template at index {idx + steps}.")
            steps += 1
        return steps



TEMPLATES_SINGLE = {
    AnaType.LineChart: Template(
        AnaType.LineChart,
        [TokenType.ANA, TokenType.FIELD, TokenType.SEP, TokenType.FIELD, TokenType.SEP],
        [Segment.OP, Segment.VAL, Segment.OP, Segment.X, Segment.OP],
        [1, 1, 1, 0, 1],
        [1, 1, 1, np.inf, 1],
        [False, False, False, True, False],
        [False, False, False, True, False],
        [(), (FieldType.String,), (), (), ()]),
    AnaType.BarChart: Template(
        AnaType.BarChart,
        [TokenType.ANA, TokenType.FIELD, TokenType.SEP, TokenType.FIELD, TokenType.SEP],
        [Segment.OP, Segment.VAL, Segment.OP, Segment.X, Segment.OP],
        [1, 1, 1, 0, 1],
        [1, 1, 1, np.inf, 1],
        [False, False, False, True, False],
        [False, False, False, True, False],
        [(), (FieldType.String,), (), (), ()]),
    AnaType.ScatterChart: Template(
        AnaType.ScatterChart,
        [TokenType.ANA, TokenType.FIELD, TokenType.SEP, TokenType.FIELD, TokenType.SEP],
        [Segment.OP, Segment.VAL, Segment.OP, Segment.X, Segment.OP],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [(), (FieldType.String,), (), (), ()]),
    AnaType.PieChart: Template(
        AnaType.PieChart,
        [TokenType.ANA, TokenType.FIELD, TokenType.SEP, TokenType.FIELD, TokenType.SEP],
        [Segment.OP, Segment.VAL, Segment.OP, Segment.X, Segment.OP],
        [1, 1, 1, 0, 1],
        [1, 1, 1, np.inf, 1],
        [False, False, False, True, False],
        [False, False, False, True, False],
        [(), (FieldType.String,), (), (), ()]),
}

TEMPLATES_MULTI = {
    AnaType.LineChart: Template(
        AnaType.LineChart,
        [TokenType.ANA, TokenType.FIELD, TokenType.SEP, TokenType.FIELD, TokenType.SEP],
        [Segment.OP, Segment.VAL, Segment.OP, Segment.X, Segment.OP],
        [1, 1, 1, 0, 1],
        [1, np.inf, 1, np.inf, 1],
        [False, False, False, True, False],
        [False, False, False, True, False],
        [(), (FieldType.String,), (), (), ()]),
    AnaType.BarChart: Template(
        AnaType.BarChart,
        [TokenType.ANA, TokenType.FIELD, TokenType.SEP, TokenType.FIELD, TokenType.SEP],
        [Segment.OP, Segment.VAL, Segment.OP, Segment.X, Segment.OP],
        [1, 1, 1, 0, 1],
        [1, np.inf, 1, np.inf, 1],
        [False, False, False, True, False],
        [False, False, False, True, False],
        [(), (FieldType.String,), (), (), ()]),
    AnaType.ScatterChart: Template(
        AnaType.ScatterChart,
        [TokenType.ANA, TokenType.FIELD, TokenType.SEP, TokenType.FIELD, TokenType.SEP],
        [Segment.OP, Segment.VAL, Segment.OP, Segment.X, Segment.OP],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [(), (FieldType.String,), (), (), ()]),
    AnaType.PieChart: Template(
        AnaType.PieChart,
        [TokenType.ANA, TokenType.FIELD, TokenType.SEP, TokenType.FIELD, TokenType.SEP],
        [Segment.OP, Segment.VAL, Segment.OP, Segment.X, Segment.OP],
        [1, 1, 1, 0, 1],
        [1, 1, 1, np.inf, 1],
        [False, False, False, True, False],
        [False, False, False, True, False],
        [(), (FieldType.String,), (), (), ()]),
    AnaType.AreaChart: Template(  # TODO: Extend GroupingOp to reflect stack/pStack/standard of area chart
        AnaType.AreaChart,
        [TokenType.ANA, TokenType.FIELD, TokenType.SEP, TokenType.FIELD, TokenType.SEP],
        [Segment.OP, Segment.VAL, Segment.OP, Segment.X, Segment.OP],
        [1, 1, 1, 0, 1],
        [1, np.inf, 1, np.inf, 1],
        [False, False, False, True, False],
        [False, False, False, True, False],
        [(), (FieldType.String,), (), (), ()]),
    AnaType.RadarChart: Template(
        AnaType.RadarChart,
        [TokenType.ANA, TokenType.FIELD, TokenType.SEP, TokenType.FIELD, TokenType.SEP],
        [Segment.OP, Segment.VAL, Segment.OP, Segment.X, Segment.OP],
        [1, 1, 1, 0, 1],
        [1, np.inf, 1, np.inf, 1],
        [False, False, False, True, False],
        [False, False, False, True, False],
        [(), (FieldType.String,), (), (), ()]),
}

TEMPLATES_MULTI_GROUPING = {
    AnaType.LineChart: Template(
        AnaType.LineChart,
        [TokenType.ANA, TokenType.FIELD, TokenType.SEP, TokenType.FIELD, TokenType.SEP],
        [Segment.OP, Segment.VAL, Segment.OP, Segment.X, Segment.OP],
        [1, 1, 1, 0, 1],
        [1, np.inf, 1, np.inf, 1],
        [False, False, False, True, False],
        [False, False, False, True, False],
        [(), (FieldType.String,), (), (), ()]),
    AnaType.BarChart: Template(
        AnaType.BarChart,
        [TokenType.ANA, TokenType.FIELD, TokenType.SEP, TokenType.FIELD, TokenType.GRP],
        [Segment.OP, Segment.VAL, Segment.OP, Segment.X, Segment.GRP],
        [1, 1, 1, 0, 1],
        [1, np.inf, 1, np.inf, 1],
        [False, False, False, True, False],
        [False, False, False, True, False],
        [(), (FieldType.String,), (), (), ()]),
    AnaType.ScatterChart: Template(
        AnaType.ScatterChart,
        [TokenType.ANA, TokenType.FIELD, TokenType.SEP, TokenType.FIELD, TokenType.SEP],
        [Segment.OP, Segment.VAL, Segment.OP, Segment.X, Segment.OP],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [(), (FieldType.String,), (), (), ()]),
    AnaType.PieChart: Template(
        AnaType.PieChart,
        [TokenType.ANA, TokenType.FIELD, TokenType.SEP, TokenType.FIELD, TokenType.SEP],
        [Segment.OP, Segment.VAL, Segment.OP, Segment.X, Segment.OP],
        [1, 1, 1, 0, 1],
        [1, 1, 1, np.inf, 1],
        [False, False, False, True, False],
        [False, False, False, True, False],
        [(), (FieldType.String,), (), (), ()]),
    AnaType.AreaChart: Template(  # TODO: Extend GroupingOp to reflect stack/pStack/standard of area chart
        AnaType.AreaChart,
        [TokenType.ANA, TokenType.FIELD, TokenType.SEP, TokenType.FIELD, TokenType.SEP],
        [Segment.OP, Segment.VAL, Segment.OP, Segment.X, Segment.OP],
        [1, 1, 1, 0, 1],
        [1, np.inf, 1, np.inf, 1],
        [False, False, False, True, False],
        [False, False, False, True, False],
        [(), (FieldType.String,), (), (), ()]),
    AnaType.RadarChart: Template(
        AnaType.RadarChart,
        [TokenType.ANA, TokenType.FIELD, TokenType.SEP, TokenType.FIELD, TokenType.SEP],
        [Segment.OP, Segment.VAL, Segment.OP, Segment.X, Segment.OP],
        [1, 1, 1, 0, 1],
        [1, np.inf, 1, np.inf, 1],
        [False, False, False, True, False],
        [False, False, False, True, False],
        [(), (FieldType.String,), (), (), ()]),
}


def prepare_templates(max_val_num: int):  # Modify templates! #VAL upper bound is limited to max_val_num.
    for templates in [TEMPLATES_MULTI, TEMPLATES_MULTI_GROUPING]:
        for template in templates.values():
            for i, segment in enumerate(template.segments):
                if segment == Segment.VAL and template.upper[i] > max_val_num:
                    template.upper[i] = max_val_num


def get_template(ana_type: AnaType, allow_multiple_values: bool, consider_grouping_operations: bool,
                 limit_search_group: bool = False):
    if allow_multiple_values:
        templates = TEMPLATES_MULTI_GROUPING if consider_grouping_operations and not limit_search_group else TEMPLATES_MULTI
    else:
        templates = TEMPLATES_SINGLE

    if ana_type not in templates:
        raise ValueError(f"{ana_type.name} template not available!")
    return templates[ana_type]
