# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from enum import Enum, IntEnum
from typing import Optional, Tuple


class TokenType(IntEnum):
    PAD = 0
    SEP = 1
    FIELD = 2
    FUNC = 3
    GRP = 4
    ANA = 5


class AnaType(Enum):
    """More details on [ANA] token. Extension of TokenType."""
    PivotTable = 0
    LineChart = 1
    BarChart = 2
    ScatterChart = 3
    PieChart = 4
    # Below 5~8 are types with few samples
    AreaChart = 5
    RadarChart = 6
    BubbleChart = 7
    SurfaceChart = 8

    @classmethod
    def from_raw_str(cls, string: str):
        if string is None:
            raise ValueError("Analysis type should not be None!")
        elif string == "pivotTable":
            return cls.PivotTable
        elif string == "lineChart" or string == "line3DChart" or string == "stockChart":
            return cls.LineChart
        elif string == "barChart" or string == "bar3DChart":
            return cls.BarChart
        elif string == "pieChart" or string == "pie3DChart" or string == "doughnutChart" or string == "ofPieChart":
            return cls.PieChart
        elif string == "scatterChart":
            return cls.ScatterChart
        # Below are types with few samples
        elif string == "areaChart" or string == "area3DChart":
            return cls.AreaChart
        elif string == "radarChart":
            return cls.RadarChart
        elif string == "bubbleChart":
            return cls.BubbleChart
        elif string == "surfaceChart" or string == "surface3DChart":
            return cls.SurfaceChart
        elif '_' in string:  # TODO: remove this
            return cls.SurfaceChart
        else:
            raise ValueError("Unexpected analysis type: {}".format(string))

    @classmethod
    def to_raw_str(cls, o):
        if not isinstance(o, cls):
            raise ValueError("Unexpected analysis type: {}".format(o))
        elif o is cls.PivotTable:
            return "pivotTable"
        elif o is cls.LineChart:
            return "lineChart"
        elif o is cls.BarChart:
            return "barChart"
        elif o is cls.ScatterChart:
            return "scatterChart"
        elif o is cls.PieChart:
            return "pieChart"
        # Below are types with few samples
        elif o is cls.AreaChart:
            return "areaChart"
        elif o is cls.RadarChart:
            return "radarChart"
        elif o is cls.BubbleChart:
            return "bubbleChart"
        elif o is cls.SurfaceChart:
            return "surfaceChart"

    @staticmethod
    def all_ana_types():
        return [ana_type for ana_type in AnaType]

    @staticmethod
    def major_chart_types():
        return [AnaType.LineChart, AnaType.BarChart, AnaType.ScatterChart, AnaType.PieChart]


class AggFunc(Enum):
    Sum = "Sum"
    Count = "Count"
    Average = "Average"
    Max = "Max"
    CountNums = "Count Numbers"
    Min = "Min"
    StdDev = "Standard deviation (samples)"
    Product = "Product"
    StdDevP = "Standard deviation (entire population)"
    Var = "Variance (samples)"
    VarP = "Variance (entire population)"

    @classmethod
    def from_raw_str(cls, string: str):
        if string is None:
            return cls.Sum
        elif string == "varp":
            return cls.VarP
        elif string == "var":
            return cls.Var
        elif string == "stdDevp":
            return cls.StdDevP
        elif string == "product":
            return cls.Product
        elif string == "stdDev":
            return cls.StdDev
        elif string == "min":
            return cls.Min
        elif string == "countNums":
            return cls.CountNums
        elif string == "max":
            return cls.Max
        elif string == "average":
            return cls.Average
        elif string == "count":
            return cls.Count
        elif string == "sum":
            return cls.Sum
        else:
            raise ValueError("Unexpected aggregation function: {}".format(string))

    @classmethod
    def to_int(cls, o):
        if not isinstance(o, cls):
            return 0
        if o is cls.Sum:
            return 1
        elif o is cls.Count:
            return 2
        elif o is cls.Average:
            return 3
        elif o is cls.Max:
            return 4
        elif o is cls.CountNums:
            return 5
        elif o is cls.Min:
            return 6
        elif o is cls.StdDev:
            return 7
        elif o is cls.Product:
            return 8
        elif o is cls.StdDevP:
            return 9
        elif o is cls.Var:
            return 10
        elif o is cls.VarP:
            return 11

    @classmethod
    def cat_num(cls, top_freq_func: int):
        return min(top_freq_func, cls.__len__()) + 1

    def int_val(self):
        return AggFunc.to_int(self)


class GroupingOp(IntEnum):
    Cluster = 1
    Stack = 2

    @classmethod
    def to_int(cls, o):
        if isinstance(o, cls):
            return int(o)
        else:
            return 0

    @classmethod
    def cat_num(cls):
        return cls.__len__() + 1

    @classmethod
    def from_raw_str(cls, string: str):
        if string is None:
            return cls.Cluster
        elif string == "standard" or string == "clustered":
            return cls.Cluster
        elif string == "stacked" or string == "percentStacked":
            return cls.Stack
        else:
            raise ValueError("Unexpected aggregation function: {}".format(string))


class Segment(IntEnum):
    """Mark the usage of each token in a sequence representing state or action space."""
    PAD = 0
    ROW = 1
    COL = 2
    X = 3
    VAL = 4
    FIELD = 5
    FUNC = 6
    GRP = 7
    OP = 8


class FieldType(IntEnum):
    Unknown = 0,
    String = 1,
    Year = 2,
    DateTime = 3,
    Decimal = 4

    @staticmethod
    def get_max():
        return max(int(val) for _, val in FieldType.__members__.items())

    @classmethod
    def from_raw_int(cls, integer: int):
        if integer == 0:
            return cls.Unknown
        elif integer == 1:
            return cls.String
        elif integer == 3:
            return cls.DateTime
        elif integer == 5:
            return cls.Decimal
        elif integer == 7:
            return cls.Year
        else:
            raise ValueError("Unexpected field type: {}".format(integer))

    @classmethod
    def cat_num(cls):
        return cls.__len__()


class FieldRole(IntEnum):
    Value = 1,
    Header = 2

    @staticmethod
    def get_max():
        return max(int(val) for _, val in FieldRole.__members__.items())

    @classmethod
    def from_raw_bool(cls, is_header: bool):
        if is_header:
            return cls.Header
        else:
            return cls.Value

    @classmethod
    def to_int(cls, o):
        if isinstance(o, cls):
            return int(o)
        else:
            return 0

    @classmethod
    def cat_num(cls):
        return cls.__len__() + 1


class IsPercent(IntEnum):
    NonPercent = 1,
    Percent = 2

    @staticmethod
    def get_max():
        return max(int(val) for _, val in IsPercent.__members__.items())

    @classmethod
    def from_raw_bool(cls, is_percent: bool):
        if is_percent:
            return cls.Percent
        else:
            return cls.NonPercent

    @classmethod
    def to_int(cls, o):
        if isinstance(o, cls):
            return int(o)
        else:
            return 0

    @classmethod
    def cat_num(cls):
        return cls.__len__() + 1


class IsCurrency(IntEnum):
    NonCurrency = 1,
    Currency = 2

    @staticmethod
    def get_max():
        return max(int(val) for _, val in IsCurrency.__members__.items())

    @classmethod
    def from_raw_bool(cls, is_currency: bool):
        if is_currency:
            return cls.Currency
        else:
            return cls.NonCurrency

    @classmethod
    def to_int(cls, o):
        if isinstance(o, cls):
            return int(o)
        else:
            return 0

    @classmethod
    def cat_num(cls):
        return cls.__len__() + 1


class HasYear(IntEnum):
    NonYear = 1,
    Year = 2

    @staticmethod
    def get_max():
        return max(int(val) for _, val in HasYear.__members__.items())

    @classmethod
    def from_raw_bool(cls, has_year: bool):
        if has_year:
            return cls.Year
        else:
            return cls.NonYear

    @classmethod
    def to_int(cls, o):
        if isinstance(o, cls):
            return int(o)
        else:
            return 0

    @classmethod
    def cat_num(cls):
        return cls.__len__() + 1


class HasMonth(IntEnum):
    NonMonth = 1,
    Month = 2

    @staticmethod
    def get_max():
        return max(int(val) for _, val in HasMonth.__members__.items())

    @classmethod
    def from_raw_bool(cls, has_month: bool):
        if has_month:
            return cls.Month
        else:
            return cls.NonMonth

    @classmethod
    def to_int(cls, o):
        if isinstance(o, cls):
            return int(o)
        else:
            return 0

    @classmethod
    def cat_num(cls):
        return cls.__len__() + 1


class HasDay(IntEnum):
    NonDay = 1,
    Day = 2

    @staticmethod
    def get_max():
        return max(int(val) for _, val in HasDay.__members__.items())

    @classmethod
    def from_raw_bool(cls, has_day: bool):
        if has_day:
            return cls.Day
        else:
            return cls.NonDay

    @classmethod
    def to_int(cls, o):
        if isinstance(o, cls):
            return int(o)
        else:
            return 0

    @classmethod
    def cat_num(cls):
        return cls.__len__() + 1


class Token:
    __slots__ = 'type', 'field_index', 'field_type', 'field_role', 'semantic_embed', 'data_features', \
                'agg_func', 'grp_op', 'ana_type', 'tags'

    def __init__(self, token_type: TokenType, field_index: Optional[int] = None,
                 field_type: Optional[FieldType] = None, field_role: Optional[FieldRole] = None,
                 semantic_embedding: Optional[np.ndarray] = None, data_characteristics: Optional[np.ndarray] = None,
                 agg_func: Optional[AggFunc] = None, grp_op: Optional[GroupingOp] = None,
                 ana_type: Optional[AnaType] = None,
                 tags: Optional[Tuple[IsPercent, IsCurrency, HasYear, HasMonth, HasDay]] = None):
        self.type = token_type
        self.semantic_embed = semantic_embedding

        # Only not None if token_type is TokenType.FIELD:
        self.field_index = field_index
        self.field_type = field_type
        self.field_role = field_role
        self.data_features = data_characteristics
        self.tags = tags

        # Only not None if token_type is TokenType.FUNC:
        self.agg_func = agg_func

        # Only not None if grp_op is TokenType.GRP:
        self.grp_op = grp_op

        # Only not None if token_type is TokenType.ANA:
        self.ana_type = ana_type
        if self.type is TokenType.ANA and self.ana_type is None:
            self.ana_type = AnaType.PivotTable  # The default analysis type

    def __lt__(self, other):
        return self.field_index < other.field_index

    def compatible_with(self, agg_func: AggFunc):
        if self.type != TokenType.FIELD:
            raise ValueError("Only can check if a field is compatible with an agg func.")

        if self.field_type is None:
            # Abandon field category features and temporarily disable compatibility check.
            return True

        if self.field_type == FieldType.String:
            # Count is the only choice for String field
            return agg_func == AggFunc.Count
        elif self.field_type in {FieldType.DateTime, FieldType.Year}:
            # Sum is invalid for datetime or year
            return agg_func != AggFunc.Sum
        else:
            return True

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Token) and self.type == o.type:
            if self.type is TokenType.ANA:
                return self.ana_type == o.ana_type
            elif self.type is TokenType.FIELD:
                return self.field_index == o.field_index
            elif self.type is TokenType.FUNC:
                return self.agg_func == o.agg_func
            elif self.type is TokenType.GRP:
                return self.grp_op == o.grp_op
            else:
                return True
        else:
            return False

    def __hash__(self) -> int:
        value = 0
        if self.type is TokenType.ANA:
            value = hash(self.ana_type)
        elif self.type is TokenType.FIELD:
            value = self.field_index
        elif self.type is TokenType.FUNC:
            value = hash(self.agg_func)
        elif self.type is TokenType.GRP:
            value = hash(self.grp_op)
        return len(TokenType) * value + self.type.value

    def __repr__(self):
        type_repr = "[%s]" % self.type.name
        if self.type is TokenType.ANA:
            return "[%s]" % self.ana_type.name
        elif self.type is TokenType.FIELD:
            return "[f%d]" % self.field_index
        elif self.type is TokenType.FUNC:
            return "[%s]" % self.agg_func.name
        elif self.type is TokenType.GRP:
            return "[%s]" % self.grp_op.name
        else:
            return type_repr
