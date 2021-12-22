import json
import logging
import os
import random
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections import namedtuple, defaultdict
from itertools import permutations
from typing import List, Optional, Tuple, Union

import numpy as np

from .config import DataConfig
from .sequence import Sequence, State
from .special_tokens import SpecialTokens
from .template import get_template
from .token import Token, TokenType, Segment, FieldType, AnaType, FieldRole, GroupingOp, \
    IsPercent, IsCurrency, HasYear, HasMonth, HasDay
from .util import get_embeddings, load_json

Measure = namedtuple('Measure', ['field', 'aggregation'])
COPY_TIMES = {"vdr": {"ja": 2, "zh": 2, "de": 2, "fr": 2, "es": 3, "en": 12},
              "pvt": {"ja": 4, "zh": 4, "de": 4, "fr": 1, "es": 1, "en": 1}}


class Field:
    __slots__ = "idx", "type", "role", "tags", "embedding", "features"

    def __init__(self, idx: int, field_type: Optional[FieldType] = None, field_role: Optional[FieldRole] = None,
                 semantic_embedding: Optional[np.ndarray] = None, data_features: Optional[np.ndarray] = None,
                 tags: Optional[Tuple[IsPercent, IsCurrency, HasYear, HasMonth, HasDay]] = None):
        self.idx = idx
        self.type = field_type
        self.role = field_role
        self.tags = tags
        self.embedding = semantic_embedding
        self.features = data_features


class Analysis(ABC):
    def __init__(self, ana_type: AnaType, aUID: str, config: DataConfig):
        self.type = ana_type
        self.aUID = aUID
        self.config = config

    def get_template(self):
        return get_template(self.type, self.config.allow_multiple_values, self.config.consider_grouping_operations,
                            self.config.limit_search_group)

    @abstractmethod
    def complete_states(self) -> List[State]:
        pass


class Chart(Analysis):
    __slots__ = "type", "aUID", "config", "x_fields", "values", "states", "grouping"

    def __init__(self, cUID: str, ana_type: AnaType, idx_to_field: dict,
                 config: DataConfig, search_sampling: bool):
        chart = load_json(config.chart_path(cUID), config.encoding)
        super().__init__(ana_type, cUID, config)
        if "yFields" in chart:
            self.values = [idx_to_field[field["index"]] for field in chart["yFields"]]
        else:
            self.values = [idx_to_field[field["index"]] for field in chart["values"]]
        if len(self.values) == 0:
            raise ValueError("No values!")
        if config.allow_multiple_values and len(self.values) > config.max_val_num:
            raise ValueError("Too many values in a chart.")

        if "xFields" in chart:
            cat_indices = [field["index"] for field in chart["xFields"]]
        else:
            cat_indices = [field["index"] for field in chart["categories"]]
        cat_indices.sort()
        for i in range(len(cat_indices) - 1):
            if cat_indices[i + 1] != cat_indices[i] + 1:
                raise ValueError("Category fields not continuous!")
        self.x_fields = [idx_to_field[index] for index in cat_indices]

        self.grouping = GroupingOp.from_raw_str(chart["grouping"]) if "grouping" in chart else None
        if ana_type is AnaType.BarChart and self.grouping is None:
            self.grouping = GroupingOp.Cluster

        # If using multiple values, we don't split the values.
        if config.allow_multiple_values:
            self.states = self.get_states(self.x_fields, self.values, self.grouping, search_sampling)
        else:
            self.states = []
            for value in self.values:
                value_states = self.get_states(self.x_fields, [value], self.grouping, search_sampling)
                self.states.extend(value_states)

    def get_states(self, x_fields: List[Token], values: List[Token],
                   grouping: Optional[GroupingOp], search_sampling: bool) -> List[State]:
        """
        :return: a list of all complete states
        """
        # Get all permutations of values and keep the permutation orders when deduplication.
        values_permutations = []
        values_set = set()
        for permutation in permutations(values):
            if permutation not in values_set:
                values_set.add(permutation)
                values_permutations.append(permutation)

        # If search_sampling, we use all value_permutations to train
        # else sample config.
        selected_states = list()
        if search_sampling:
            for values_perm in values_permutations:
                selected_states.append(self.get_state(x_fields, values_perm, grouping))
        else:
            # Get a fixed set of random indexes (by set a fixed random seed)
            num_permute_samples = min(self.config.num_permute_samples[len(values)], len(values_permutations))
            random.seed(1003)
            # Make sure the order of permutation (1st is the original order)
            selected_indexes = {0}  # Make sure that the permutation with the origin order is selected
            selected_indexes.update(random.sample(list(range(1, len(values_permutations))), k=num_permute_samples - 1))

            # Apply the random indexes to value_permutation.
            # An example:
            #   origin values are (1,2,3)
            #   permutations are (1,2,3), (1,3,2), (2,1,3), (2,3,1), (3,1,2), (3,2,1)
            #   selected_indexes are [4,1,3]
            #   Then we'll take perm[0,1,3], i.e. (1,2,3), (1,3,2), (2,3,1) into our dataset.
            for perm_idx in sorted(selected_indexes):
                values_perm = values_permutations[perm_idx]
                selected_states.append(self.get_state(x_fields, values_perm, grouping))

        return selected_states

    def get_state(self, x_fields: List, values: List, grouping: Optional[GroupingOp]) -> State:
        state = State.fill_template(self.get_template(), {
            Segment.X: x_fields,
            Segment.VAL: values,
            Segment.GRP: None if grouping is None else [SpecialTokens.get_grp_token(grouping)]
        })
        return state

    def seq_len(self) -> int:
        if not hasattr(self, "states") or len(self.states) == 0:
            return 0
        return len(self.states[0])

    def complete_states(self) -> List[State]:
        return self.states


def table_fields(info: Union[str, dict], field_dicts: List[dict], config: DataConfig) -> List[Field]:
    fields = [Field(fd["index"]) for fd in field_dicts]
    s_id = ".".join(info.split(".")[:-1]) if isinstance(info, str) else None

    # Get the header title embedding of each field
    if config.use_semantic_embeds:
        idx = 0
        for fd, embed in zip(field_dicts,
                             get_embeddings(s_id, config) if isinstance(info, str) else info["embeddings"]):
            fields[idx].embedding = embed[config.embed_layer][config.embed_reduce_type]
            idx += 1

    # Get the data characteristics of each field
    if config.use_data_features:
        for idx, fd in enumerate(field_dicts):
            fields[idx].features = np.array(config.data_cleanup_fn(fd["dataFeatures"]))

    # Get categorical features of each field
    for idx, fd in enumerate(field_dicts):
        if config.use_field_type:
            fields[idx].type = FieldType.from_raw_int(fd["type"])
        if config.use_field_role:
            fields[idx].role = FieldRole.from_raw_bool(fd["inHeaderRegion"])
        if config.use_binary_tags:
            fields[idx].tags = (
                IsPercent.from_raw_bool(fd["isPercent"]),
                IsCurrency.from_raw_bool(fd["isCurrency"]),
                HasYear.from_raw_bool(fd["hasYear"]),
                HasMonth.from_raw_bool(fd["hasMonth"]),
                HasDay.from_raw_bool(fd["hasDay"])
            )

    return fields


def generate_action_space(fields: List[Field], special_tokens: SpecialTokens,
                          consider_grouping_operations: bool, top_freq_func: int):
    """Generate the whole action space for a source."""
    tokens = [Token(TokenType.FIELD, field_index=field.idx, field_type=field.type, field_role=field.role,
                    semantic_embedding=field.embedding, data_characteristics=field.features, tags=field.tags)
              for field in fields]
    if consider_grouping_operations:
        grp_tokens = special_tokens.GRP_OP_TOKENS
        grp_segments = [Segment.GRP] * len(grp_tokens)
    else:
        grp_tokens = grp_segments = []
    action_space = Sequence(
        tokens + [SpecialTokens.SEP_TOKEN] + grp_tokens + special_tokens.AGG_FUNC_TOKENS[:top_freq_func],
        [Segment.FIELD] * len(tokens) + [Segment.OP] + grp_segments + [Segment.FUNC] * top_freq_func)
    idx2field = {t.field_index: t for t in tokens}

    # Final check the fields_index is as expected
    idx = 0
    for t in tokens:
        assert t.field_index == idx, "Field index should be its actual 0-based index."
        idx += 1
    return action_space, idx2field


class DataTable:
    __slots__ = "config", "tUID", "pUIDs", "cUIDs", "cTypes", "ana_type_set", \
                "n_rows", "n_cols", "action_space", "idx2field"

    def __init__(self, info: Union[str, dict], special_tokens: SpecialTokens, config: DataConfig):
        self.config = config
        if isinstance(info, dict):  # Don't have ground truth
            table = info["table"]
            self.tUID = None
            self.n_cols = None
        else:
            table = load_json(config.table_path(info), config.encoding)
            sUID = ".".join(info.split(".")[:-1])  # Because there may be '.' in Plotly's uid.
            table_id = info.split(".")[-1][1:]

            # TODO: A more efficient way that only load sample.json once for each schema. -- workItems 51
            # Load down-sampling file, and get selected chart UIDs and types.
            ana_list = load_json(config.sample_path(sUID), config.encoding)["tableAnalysisPairs"][str(table_id)]
            cTypes, cUIDs, pUIDs = [], [], []
            for i, ana_info in enumerate(ana_list):
                ana_type = AnaType.from_raw_str(ana_info['anaType'])
                if ana_type == AnaType.PivotTable:
                    # a pivot table
                    if config.max_dim_num < 0 or (config.max_dim_num >= ana_info['nColDim'] and
                                                  config.max_dim_num >= ana_info['nRowDim']):
                        pUIDs.append(f"{info}.p{i}")
                else:
                    # a chart
                    if ana_info['nVals'] > 0 and \
                            (not config.allow_multiple_values or (config.max_val_num >= ana_info['nVals'])):
                        cTypes.append(ana_type)
                        cUIDs.append(f"{info}.c{i}")

            self.tUID = info
            self.pUIDs = pUIDs
            self.cUIDs = cUIDs  # chart id
            self.cTypes = cTypes  # chart type

            self.ana_type_set = set(self.cTypes)
            if len(self.pUIDs) > 0:
                self.ana_type_set.add(AnaType.PivotTable)
            self.n_rows = table["nRows"]
            self.n_cols = table["nColumns"]

        # Generate the action space of the table
        fields = table_fields(info, table['fields'], config)
        # Chart-only data loading have no agg_func action space (config.top_freq_func is set to zero)
        self.action_space, self.idx2field = generate_action_space(fields, special_tokens,
                                                                  config.consider_grouping_operations,
                                                                  config.top_freq_func)


class Index:
    # TODO: Support different ways to load and query index.
    # Such as only keep tables that contains analysis types specified by the config.
    def __init__(self, config: DataConfig):
        logger = logging.getLogger(f"Index init()")

        self.config = config
        # After down sampling, the index information can be very huge, and stored in 1 single file
        with open(config.index_path(), "r", encoding=config.encoding) as f:
            self.index = json.load(f, object_pairs_hook=OrderedDict)

        self.total_files = len(self.index)  # here total files = total schema
        if config.load_at_most:
            self.total_files = min(self.total_files, config.load_at_most)

        # Initialize f_end and tUIDs. Every file type has its own list.
        self.f_end = []  # table end index in self.tUIDs of current schema
        self.tUIDs = []  # [schema_id*.t*, ]
        self.ana_type_idx = defaultdict(list)  # ana_type to List of tables that contain the analysis type.
        self.langs = []  # language of each table
        self.datasets = []  # dataste of each table

        self.filter_by_openfile = 0
        self.filter_by_lang = 0
        self.filter_by_no_embedding = 0
        self.filter_by_too_many_fields = 0
        self.filter_by_no_valid_analysis = 0
        self.pivot_filter_by_too_many_dimension = 0
        self.chart_filter_by_no_valnum = 0
        self.chart_filter_by_too_many_values = 0
        total_charts = 0
        total_pivots = 0

        for schema_index in self.index:
            try:
                with open(config.sample_path(schema_index), "r", encoding=config.encoding) as f:
                    sampled_schema = json.load(f)
            except Exception:
                self.filter_by_openfile += 1
                continue

            # schema_id = sampled_schema['sID']
            lang = sampled_schema['lang']
            if lang == "zh_cht" or lang == "zh_chs":
                lang = "zh"
            schema_tUIDs = []
            schema_tUID_indices = defaultdict(list)
            schema_chart_types = set()
            # TODO: idx = len(schema_tUIDs) - 1

            if config.use_semantic_embeds and not os.path.exists(config.embedding_path(schema_index)):
                # bypass embedding file not found error
                self.filter_by_no_embedding += 1
                continue

            if not config.has_language(lang):
                self.filter_by_lang += 1
                continue

            if 'bing' in sampled_schema:
                self.tUIDs.append(f"{schema_index}.t0")
                self.f_end.append(len(self.tUIDs))
                self.langs.append(lang)
                continue

            for table_id, ana_info_list in sampled_schema['tableAnalysisPairs'].items():
                # ana_info_list: List[Dict]
                if sampled_schema['nColumns'] > config.max_field_num:
                    self.filter_by_too_many_fields += 1
                    continue
                tUID = f"{schema_index}.t{table_id}"

                # TODO: remove this bypass logic. (Work item #63)
                if not os.path.exists(config.table_path(tUID)):
                    continue

                ana_type_list = []
                for i, ana_info in enumerate(ana_info_list):
                    ana_type = AnaType.from_raw_str(ana_info['anaType'])
                    if ana_type == AnaType.PivotTable:
                        # a pivot table analysis
                        if config.max_dim_num < 0 or (config.max_dim_num >= ana_info['nColDim'] and
                                                      config.max_dim_num >= ana_info['nRowDim']):
                            ana_type_list.append(ana_type)
                            total_pivots += 1
                        else:
                            self.pivot_filter_by_too_many_dimension += 1
                    else:
                        # a chart analysis, params is a int which is #value
                        if ana_info['nVals'] == 0:
                            self.chart_filter_by_no_valnum += 1
                        elif config.allow_multiple_values and config.max_val_num < ana_info['nVals']:
                            self.chart_filter_by_too_many_values += 1
                        elif not config.allow_multiple_values or (config.max_val_num >= ana_info['nVals']):
                            ana_type_list.append(ana_type)
                            total_charts += 1

                if len(ana_type_list) > 0:
                    schema_tUIDs.append(tUID)
                else:
                    self.filter_by_no_valid_analysis += 1
                    continue

                for cur_ana_type in ana_type_list:
                    if cur_ana_type != AnaType.PivotTable:
                        schema_chart_types.add(cur_ana_type)
                for cur_ana_type in filter(lambda t: t in ana_type_list, config.input_types):
                    schema_tUID_indices[cur_ana_type].append(len(schema_tUIDs) - 1)

            # Merge the table and analysis information of the current schema into the whole schema records.
            for ana_type in AnaType:  # TODO: comments
                self.ana_type_idx[ana_type].extend(
                    map(lambda x: x + len(self.tUIDs), schema_tUID_indices[ana_type]))
            self.tUIDs.extend(schema_tUIDs)
            self.f_end.append(len(self.tUIDs))
            self.langs.extend([lang] * len(schema_tUIDs))

        logger.info(f"Total schemas is {len(self.index)}")
        logger.info(f"file filtered by language: {self.filter_by_lang}")
        logger.info(f"file filtered by openfile: {self.filter_by_openfile}")
        logger.info(f"table filtered by no embedding: {self.filter_by_no_embedding}")
        logger.info(f"table filtered by too many fields: {self.filter_by_too_many_fields}")
        logger.info(f"table filtered by no valid analysis: {self.filter_by_no_valid_analysis}")
        logger.info(f"chart filtered by no valnum: {self.chart_filter_by_no_valnum}")
        logger.info(f"chart filtered by too many values: {self.chart_filter_by_too_many_values}")
        logger.info(f"pivot filtered by too many dimensions: {self.pivot_filter_by_too_many_dimension}")
        logger.info(f"Total tables is {len(self.tUIDs)}")
        logger.info(f"Total charts is {total_charts}")
        logger.info(f"Total pivot tables is {total_pivots}")

    def train_tUIDs(self):
        train_threshold = self.config.train_ratio * len(self.f_end)
        if train_threshold < len(self.f_end):
            end_index = self.f_end[int(train_threshold)]
        else:
            end_index = self.f_end[-1]
        return self.get_tUIDs(0, end_index)

    def valid_tUIDs(self):
        train_threshold = self.config.train_ratio * len(self.f_end)
        valid_threshold = (self.config.train_ratio + self.config.valid_ratio) * len(self.f_end)
        if train_threshold < len(self.f_end):
            start_index = self.f_end[int(train_threshold)]
        else:
            start_index = self.f_end[-1]
        if valid_threshold < len(self.f_end):
            end_index = self.f_end[int(valid_threshold)]
        else:
            end_index = self.f_end[-1]
        return self.get_tUIDs(start_index, end_index)

    def test_tUIDs(self):
        valid_threshold = (self.config.train_ratio + self.config.valid_ratio) * len(self.f_end)
        if valid_threshold < len(self.f_end):
            start_index = self.f_end[int(valid_threshold)]
        else:
            start_index = self.f_end[-1]
        end_index = self.f_end[-1]
        return self.get_tUIDs(start_index, end_index)

    def get_tUIDs(self, start_index=None, end_index=None):
        '''
        For every analysis type in config, the corresponding tUIDs within [start_index, end_index)
        will be extracted and concatenated as a big list.
        :param lang: Language that the extracted tUIDs should be.
        :param dataset: The type of dataset that the extracted tUIDs should come from.
        :param copy: Copy specific times of specific tUIDs if copy is True.
        '''

        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = len(self.tUIDs)

        indices = []
        for ana_type in self.config.input_types:
            indices_of_type = list(
                filter(lambda idx: start_index <= idx < end_index, self.ana_type_idx[ana_type]))
            indices.extend(indices_of_type)
        indices = list(set(indices))

        # For empirical study, we use all tables.
        if self.config.empirical_study:
            indices = list(range(start_index, end_index))

        indices.sort()
        tUIDs = [self.tUIDs[idx] for idx in indices]
        return tUIDs

    def save_dataset_split(self):
        logger = logging.getLogger(f"Index save_dataset_split()")
        with open(os.path.join(self.config.corpus_path, "train.txt"), 'w', encoding="utf-8-sig") as f:
            train_tuids = self.train_tUIDs()
            json.dump(train_tuids, f)
        with open(os.path.join(self.config.corpus_path, "valid.txt"), 'w', encoding="utf-8-sig") as f:
            valid_tuids = self.valid_tUIDs()
            json.dump(valid_tuids, f)
        with open(os.path.join(self.config.corpus_path, "test.txt"), 'w', encoding="utf-8-sig") as f:
            test_tuids = self.test_tUIDs()
            json.dump(test_tuids, f)
        logger.info("train/valid/test tUIDs saved.")
