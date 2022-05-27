# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from os import path
from typing import Optional, List

from .token import AggFunc, AnaType, FieldType, FieldRole, GroupingOp, IsPercent, IsCurrency, HasYear, HasMonth, HasDay

EMBED_MODELS = [
    "bert-base-uncased",
    "bert-base-cased",
    "bert-large-uncased",
    "bert-large-cased",
    "bert-base-multilingual-cased",
    "glove.6B.50d",
    "glove.6B.100d",
    "glove.6B.200d",
    "glove.6B.300d",
    "fasttext",
    "xlm-roberta-base"
]

MODEL_LANG = {
    "bert-base-uncased": "en",
    "bert-base-cased": "en",
    "bert-large-uncased": "en",
    "bert-large-cased": "en",
    "bert-base-multilingual-cased": "mul",
    "glove.6B.50d": "en",
    "glove.6B.100d": "en",
    "glove.6B.200d": "en",
    "glove.6B.300d": "en",
    "fasttext": "mul",
    "xlm-roberta-base": "mul"
}

EMBED_LEN = {
    "bert-base-uncased": 768,
    "bert-base-cased": 768,
    "bert-large-uncased": 1024,
    "bert-large-cased": 1024,
    "bert-base-multilingual-cased": 768,
    "glove.6B.50d": 50,
    "glove.6B.100d": 100,
    "glove.6B.200d": 200,
    "glove.6B.300d": 300,
    "fasttext": 50,
    "xlm-roberta-base": 768
}

EMBED_LAYERS = {
    "bert-base-uncased": {-2, -12},
    "bert-base-cased": {-2, -12},
    "bert-large-uncased": {-2, -12},
    "bert-large-cased": {-2, -12},
    "bert-base-multilingual-cased": {-2, -12},
    "glove.6B.50d": {0},
    "glove.6B.100d": {0},
    "glove.6B.200d": {0},
    "glove.6B.300d": {0},
    "fasttext": {0},
    "xlm-roberta-base": {-2, -12}
}

DF_FEATURE_NUM = 31

FEATURE_MAP = {
    'aggrPercentFormatted': 1,
    'aggr01Ranged': 2,
    'aggr0100Ranged': 3,  # Proportion of values ranged in 0-100
    'aggrIntegers': 4,  # Proportion of integer values
    'aggrNegative': 5,  # Proportion of negative values
    'commonPrefix': 6,  # Proportion of most common prefix digit
    'commonSuffix': 7,  # Proportion of most common suffix digit
    'keyEntropy': 8,  # Entropy by values
    'charEntropy': 9,  # Entropy by digits/chars
    'range': 10,  # Values range
    'changeRate': 11,  # Proportion of different adjacent values
    'partialOrdered': 12,  # Maximum proportion of increasing or decreasing adjacent values
    'variance': 13,  # Standard deviation
    'cov': 14,  # Coefficient of variation
    'cardinality': 15,  # Proportion of distinct values
    'spread': 16,  # Cardinality divided by range
    'major': 17,  # Proportion of the most frequent value
    'benford': 18,  # Distance of the first digit distribution to real-life average
    'orderedConfidence': 19,  # Indicator of sequentiality
    'equalProgressionConfidence': 20,  # confidence for a sequence to be equal progression
    'geometircProgressionConfidence': 21,  # confidence for a sequence to be geometric progression
    'medianLength': 22,  # median length of fields' records, 27.5 is 99% value
    'lengthVariance': 23,  # transformed length stdDev of a sequence
    'sumIn01': 24,
    'sumIn0100': 25,
    'absoluteCardinality': 26,
    'skewness': 27,
    'kurtosis': 28,
    'gini': 29,
    'nRows': 30,
    'averageLogLength': 31
}
INDEX_FEATURE = {v: k for k, v in FEATURE_MAP.items()}

TYPE_MAP = {
    FieldType.Unknown: 0,
    FieldType.String: 1,
    FieldType.DateTime: 3,
    FieldType.Decimal: 5,
    FieldType.Year: 7
}


def cleanup_data_features_nn(data_features: dict):
    """
    Clean up data features that used in neural network models.
    Features like 'range', 'variance', 'cov', 'lengthStdDev' are in range [-inf, inf] or [0, inf].
    These features may cause problems in NN model and need to be normalized.
    We adopt normalization by distribution here.
    To take range as an example, this feature distributes in [0, inf]. We first square root this feature.
    Then examining the distribution (CDF) of the feature, we find that 99% square-rooted values less than 25528.5.
    Therefore, we normalize it by 25528.5. If the value is greater than this threshold (25528.5), they are set to 1.
    """
    # Normalize range, var and cov
    raw_range = data_features.get('range', 0.0)
    norm_range = 1 if isinstance(raw_range, str) else min(1.0, math.sqrt(raw_range) / 25528.5)
    raw_var = data_features.get('variance', 0.0)
    norm_var = 1 if isinstance(raw_var, str) else min(1.0, math.sqrt(raw_var) / 38791.2)
    raw_cov = data_features.get('cov', 0.0)
    if isinstance(raw_cov, str):
        norm_cov = 1
    else:
        norm_cov = min(1.0, math.sqrt(raw_cov) / 55.2) if raw_cov >= 0 else \
            max(-1.0, -1.0 * math.sqrt(abs(raw_cov)) / 633.9)
    # Use standard deviation rather than variance of feature 'lengthVariance'
    # 99% length stdDev of fields' records is less than 10
    lengthStdDev = min(1.0, math.sqrt(data_features.get('lengthVariance', 0.0)) / 10.0)

    # There are NAN or extremely large values in skewness and kurtosis, so we set:
    # skewness: NAN -> 0.0, INF/large values -> 1.0
    # kurtosis: NAN -> 0.0, INF/large values -> 1.0
    # skewness 99%ile = 3.844
    # kurtosis 99%ile = 0.7917 (no normalization)
    skewness_99ile = 3.844
    skewness = data_features.get('skewness', 0.0)
    if skewness == "NAN":
        skewness = 0.0
    elif isinstance(skewness, str) or abs(skewness) > skewness_99ile:
        skewness = skewness_99ile
    skewness = skewness / skewness_99ile

    kurtosis = data_features.get('kurtosis', 0.0)
    if kurtosis == "NAN":
        kurtosis = 0.0
    elif isinstance(kurtosis, str) or abs(kurtosis) > 1.0:
        kurtosis = 1.0

    gini = data_features.get('gini', 0.0)
    if gini == "NAN":
        gini = 0.0
    elif isinstance(gini, str) or abs(gini) > 1.0:
        gini = 1.0

    benford = data_features.get('benford', 0.0)
    if benford == "NAN":
        benford = 0.0
    elif isinstance(benford, str) or abs(benford) > 1.036061:
        benford = 1.036061

    features = [
        data_features.get('aggrPercentFormatted', 0),  # Proportion of cells having percent format
        data_features.get('aggr01Ranged', 0),  # Proportion of values ranged in 0-1
        data_features.get('aggr0100Ranged', 0),  # Proportion of values ranged in 0-100
        data_features.get('aggrIntegers', 0),  # Proportion of integer values
        data_features.get('aggrNegative', 0),  # Proportion of negative values
        data_features['commonPrefix'],  # Proportion of most common prefix digit
        data_features['commonSuffix'],  # Proportion of most common suffix digit
        data_features['keyEntropy'],  # Entropy by values
        data_features['charEntropy'],  # Entropy by digits/chars
        norm_range,  # data_features.get('range', 0),  # Values range
        data_features['changeRate'],  # Proportion of different adjacent values
        data_features.get('partialOrdered', 0),  # Maximum proportion of increasing or decreasing adjacent values
        norm_var,  # data_features.get('variance', 0),  # Standard deviation
        norm_cov,  # data_features.get('cov', 0),  # Coefficient of variation
        data_features['cardinality'],  # Proportion of distinct values
        data_features.get('spread', 0),  # Cardinality divided by range
        data_features['major'],  # Proportion of the most frequent value
        benford,  # Distance of the first digit distribution to real-life average
        data_features.get('orderedConfidence', 0),  # Indicator of sequentiality
        data_features.get('equalProgressionConfidence', 0),  # confidence for a sequence to be equal progression
        data_features.get('geometircProgressionConfidence', 0),  # confidence for a sequence to be geometric progression
        min(1, data_features.get('medianLength', 0) / 27.5),  # median length of fields' records, 27.5 is 99% value
        lengthStdDev,  # transformed length stdDev of a sequence
        data_features.get('sumIn01', 0.0),  # Sum the values when they are ranged 0-1
        data_features.get('sumIn0100', 0.0) / 100,  # Sum the values when they are ranged 0-100
        min(1, data_features.get('absoluteCardinality', 0.0) / 344),  # Absolute Cardinality, 344 is 99% value
        skewness,
        kurtosis,
        gini,
        data_features.get('nRows', 0.0) / 576,  # Number of rows, 576 is 99% value
        data_features.get('averageLogLength', 0.0)
    ]
    for i, f in enumerate(features):
        if isinstance(f, str) or abs(f) > 10000:
            print("WARNING: feature[{}] is {}".format(i, f))
    return [0 if isinstance(f, str) else f for f in features]


class DataConfig:
    """Data configurations to specify data loading and representation formats"""

    def __init__(self, corpus_path: str = "/storage/chart-20200830/", encoding: str = "utf-8-sig",
                 max_field_num: int = 128, max_val_num: int = 4, max_dim_num: int = 4,
                 unified_ana_token: bool = False, top_freq_func: Optional[int] = None, allow_agg_func: bool = True,
                 use_field_type: bool = True, use_field_role: bool = True, use_binary_tags: bool = True,
                 use_data_features: bool = True, use_semantic_embeds: bool = True,
                 allow_multiple_values: bool = False, consider_grouping_operations: bool = True,
                 need_field_indices: bool = False, field_permutation: bool = False,
                 embed_model: str = "bert-base-multilingual-cased",
                 embed_format: str = "pickle", embed_layer: int = -2, embed_reduce_type: str = "mean",
                 train_ratio: float = 0.7, valid_ratio: float = 0.1, test_ratio: float = 0.2,
                 num_train_analysis: int = None, load_at_most: Optional[int] = None, empirical_study: bool = False,
                 search_types: Optional[List[AnaType]] = None, searching_all_type: bool = False,
                 model_types: Optional[List[AnaType]] = None, input_types: Optional[List[AnaType]] = None,
                 lang: str = 'en', limit_search_group: bool = False):
        """
        :param corpus_path: the root dir of the corpus
        :param encoding: default json encoding
        :param english_only: load only "en" schemas?
        :param max_field_num: the max number of fields allowed in a table, only has effect when allow_multiple_values
        :param max_val_num: the max number of values/series in a chart, only useful when allow_multiple_values
        :param max_dim_num: the max number of field as column/row dimension in a pivot table.
        :param unified_ana_token: if all analysis types share a same AnaType (One TokenType.ANA)
        :param top_freq_func: the number of top frequently used aggregation functions to consider as actions,
        will be set to 0 if search_types does not contain PivotTable
        :param allow_agg_func: if top_freq_func is used as a categorical feature of a token
        :param use_field_type: if FieldType is used as a categorical feature of a token
        :param use_field_role: if FieldRole is used as a categorical feature of a token
        :param use_data_features: if data characteristics vector is used as part of a token
        :param use_semantic_embeds: if semantic header embedding vector is used as part of a token
        :param allow_multiple_values: shall we allow multiple values in one analysis sequence
        and generate its permutations, or split a multi-value chart into multiple single-value ones
        :param consider_grouping_operations: shall we differentiate clustered and stacked Bar charts by a new operator
        :param need_field_indices: do model need original table field index as input (useful when field_permutation)
        :param field_permutation: whether to permute fields (in action space) during training
        :param embed_model: which type of header embedding to adopt
        :param embed_format: "json" or "pickle"
        :param embed_layer: choose a layer from EMBED_LAYERS wrt embed_model
        :param embed_reduce_type: "mean" or "max"
        :param num_train_analysis: number of Analysis taken when training
        :param search_types: List[AnaType] indicating what (table, analysis) pairs are searched in beam searching agents.
        By default this will include all analysis types if None is given.
        :param searching_all_type: Searching by start with all types in search_types (otherwise only ground truths)
        :param model_types: List[AnaType] indicating how a previous model was created and trained.
        Need to match the previously trained model for proper model loading.
        By default this is will be search_types if None is given.
        :param input_types: List[AnaType] indicating what (table, analysis) pairs are loaded from corpus.
        Analysis types not in the list will be filtered out.
        By default this is will be search_types if None is given.
        :param lang: only keep the tables with headers in the specified language(s).
        :param limit_search_group: If it's True, not search Group.(We don't have Group in Plotly).
        """
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.lang = lang
        self.english_only = (self.lang == "en")

        self.max_field_num = max_field_num
        self.max_val_num = max_val_num
        self.max_dim_num = max_dim_num
        self.need_field_indices = need_field_indices
        self.field_permutation = field_permutation
        self.unified_ana_token = unified_ana_token
        self.allow_multiple_values = allow_multiple_values
        self.limit_search_group = limit_search_group

        self.use_field_type = use_field_type
        self.use_field_role = use_field_role
        self.use_binary_tags = use_binary_tags
        self.use_data_features = use_data_features
        self.use_semantic_embeds = use_semantic_embeds

        self.load_at_most = load_at_most
        self.num_train_analysis = num_train_analysis

        if embed_model not in EMBED_MODELS:
            raise ValueError("{} is not a valid model name.".format(embed_model))
        self.embed_model = embed_model
        model_lang = MODEL_LANG[embed_model]
        if model_lang != "mul" and not self.english_only:
            raise ValueError("Model language is {} while english_only = {}".format(model_lang, self.english_only))
        self.embed_len = EMBED_LEN[embed_model] if use_semantic_embeds else 0

        if embed_format == "pickle":
            self.embed_in_json = False
        elif embed_format == "json":
            self.embed_in_json = True
        else:
            raise ValueError("Embedding format {} is unrecognizable.".format(embed_format))

        if embed_layer in EMBED_LAYERS[embed_model]:
            self.embed_layer = str(embed_layer) if "fast" in embed_model else embed_layer

        else:
            raise ValueError("Embedding layer {} not available for model {}".format(embed_layer, embed_model))

        if embed_reduce_type == "mean" or embed_reduce_type == "max":
            self.embed_reduce_type = embed_reduce_type
        else:
            raise ValueError("Embedding type {} is unrecognizable.".format(embed_reduce_type))

        self.data_len = DF_FEATURE_NUM if use_data_features else 0
        self.data_cleanup_fn = cleanup_data_features_nn

        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.empirical_study = empirical_study

        self.cat_nums = []  # Notice: The order in categories impacts sequence.to_dict() !!
        if use_field_type:
            self.cat_nums.append(FieldType.cat_num())
        if use_field_role:
            self.cat_nums.append(FieldRole.cat_num())
        if use_binary_tags:
            self.cat_nums.append(IsPercent.cat_num())
            self.cat_nums.append(IsCurrency.cat_num())
            self.cat_nums.append(HasYear.cat_num())
            self.cat_nums.append(HasMonth.cat_num())
            self.cat_nums.append(HasDay.cat_num())

        self.search_all_types = searching_all_type
        if search_types is None:
            search_types = AnaType.all_ana_types()
            self.search_all_types = True
        self.search_types = search_types
        if model_types is None:
            self.model_types = self.search_types
        else:
            self.model_types = model_types
        if input_types is None:
            self.input_types = self.search_types
        else:
            self.input_types = input_types

        # Check if model_types and search_types are compatible
        # to ensure the action space of the model works for search types.
        difference = set(self.search_types) - set(self.model_types)
        if AnaType.BarChart in difference or AnaType.PivotTable in difference:
            # TODO: here we assume consider_grouping_operations is the same for previous model and current run
            raise ValueError(f"Incompatible search_types and model_types. Diff: {difference}")

        if AnaType.BarChart not in self.model_types:
            consider_grouping_operations = False
        self.consider_grouping_operations = consider_grouping_operations
        if consider_grouping_operations:
            self.cat_nums.append(GroupingOp.cat_num())

        if AnaType.PivotTable in self.model_types and allow_agg_func:
            self.top_freq_func = len(AggFunc) if top_freq_func is None else top_freq_func
        else:
            self.top_freq_func = 0

        if self.top_freq_func > 0:
            self.cat_nums.append(AggFunc.cat_num(self.top_freq_func))  # See AggFunc.to_int()
        self.cat_len = len(self.cat_nums)

        self.empty_embed = [0.] * self.embed_len
        self.empty_cat = [0] * self.cat_len
        self.empty_data = [0.] * self.data_len

        self.num_permute_samples = [0, 1, 2, 4, 8]

    def has_language(self, language: str):
        if self.lang == "mul" or language == self.lang:
            return True
        return False

    def num_cmd_tokens(self):
        # Command tokens: [SEP], GroupingOperations and AggFunctions
        return 1 + (len(GroupingOp) if self.consider_grouping_operations else 0) + self.top_freq_func

    def index_path(self):
        # return path.join(self.corpus_path, "index", "merged-unique.json")
        return path.join(self.corpus_path, "index", "schema_ids.json")

    def sample_path(self, sUID: str):
        return path.join(self.corpus_path, "sample-new", f"{sUID}.sample.json")

    def file_info_path(self, fUID: str):
        return path.join(self.corpus_path, "data", f"{fUID}.json")

    def table_path(self, tUID: str):
        return path.join(self.corpus_path, "data", f"{tUID}.DF.json")

    def vdr_table_path(self, tUID: str):
        return path.join(self.corpus_path, "data", f"{tUID}.json")

    def mutual_info_path(self, tUID: str):
        return path.join(self.corpus_path, "data", f"{tUID}.MI.json")

    def embedding_path(self, uID: str):
        if self.embed_in_json:
            return path.join(self.corpus_path, "embeddings", self.embed_model, f"{uID}.EMB.json")
        else:
            return path.join(self.corpus_path, "embeddings", self.embed_model, f"{uID}.pickle")

    def pivot_table_path(self, pUID: str):
        return path.join(self.corpus_path, "data", f"{pUID}.json")

    def chart_path(self, cUID: str):
        return path.join(self.corpus_path, "data", f"{cUID}.json")


DEFAULT_LANGUAGES = ["en", "mul"]

# Feature choices "ablation-embedding[-(single|x_group)]"
DEFAULT_FEATURE_CHOICES = ["all-en_bert", "all-fast", "all-glove", "all-mul_bert",
                           "all-en_bert-x_group", "all-fast-x_group",
                           "all-en_bert-single", "all-fast-single",
                           "embed-mul_bert", "embed-en_bert", "embed-glove",
                           "data-fast", "type-fast", "cat-fast", "embed-fast",
                           "nodata-fast", "nocat-fast", "noembed-fast",
                           "metadata-mul_bert", "metadata-fast-single", "metadata-turing"]

# Analysis types for Table2Analysis/Charts
DEFAULT_ANALYSIS_TYPES = [AnaType.to_raw_str(ana_type) for ana_type in AnaType] + ["all", "allCharts"]


def convert_ana_types(type_str: str):
    if type_str is None:
        return None
    elif type_str == "all":
        return AnaType.all_ana_types()
    elif type_str == "allCharts":
        return AnaType.major_chart_types()
    else:
        return [AnaType.from_raw_str(type_str)]


def get_data_config(corpus_path: str, constraint: str, search_types_str: Optional[str] = None,
                    previous_types_str: Optional[str] = None, input_types_str: Optional[str] = None,
                    unified_ana_token: bool = False, num_train_analysis: int = None,
                    field_permutation: bool = False, lang: str = 'en', mode: str = None,
                    empirical_study: bool = False, limit_search_group: bool = False):
    # TODO: put unified_ana_token and field_permutation into constraint str
    configs = constraint.split('-')

    if configs[0] == 'all':
        use_data_features = True
        use_semantic_embeds = True
        use_field_type = True
        use_field_role = True
        use_binary_tags = True
        allow_agg_func = True
    elif configs[0] == 'type':
        use_data_features = False
        use_semantic_embeds = False
        use_field_type = True
        use_field_role = True
        use_binary_tags = False
        allow_agg_func = True
    elif configs[0] == 'data':  # only data features
        use_data_features = True
        use_semantic_embeds = False
        use_field_type = False
        use_field_role = False
        use_binary_tags = False
        allow_agg_func = True
    elif configs[0] == 'nodata':  # no data features
        use_data_features = False
        use_semantic_embeds = True
        use_field_type = True
        use_field_role = True
        use_binary_tags = True
        allow_agg_func = True
    elif configs[0] == 'cat':  # only categorical features
        use_data_features = False
        use_semantic_embeds = False
        use_field_type = True
        use_field_role = True
        use_binary_tags = True
        allow_agg_func = True
    elif configs[0] == 'nocat':  # no categorical features
        use_data_features = True
        use_semantic_embeds = True
        use_field_type = False
        use_field_role = False
        use_binary_tags = False
        allow_agg_func = True
    elif configs[0] == 'embed':  # only header embedding
        use_data_features = False
        use_semantic_embeds = True
        use_field_type = False
        use_field_role = False
        use_binary_tags = False
        allow_agg_func = True
    elif configs[0] == 'noembed':  # no header embedding
        use_data_features = True
        use_semantic_embeds = False
        use_field_type = True
        use_field_role = True
        use_binary_tags = True
        allow_agg_func = True
    elif configs[0] == 'metadata':
        use_data_features = True
        use_semantic_embeds = True
        use_field_type = True
        use_field_role = False
        use_binary_tags = False
        allow_agg_func = False
    else:
        raise NotImplementedError(f"Data config for {configs[0]} not yet implemented.")

    if len(configs) == 1:
        configs.append('fast')
    if configs[1] == 'en_bert':
        embed_model = "bert-base-uncased"
        embed_format = "pickle"
        embed_layer = -2
        embed_reduce_type = "mean"
    elif configs[1] == 'mul_bert':
        embed_model = "bert-base-multilingual-cased"
        embed_format = "pickle"
        embed_layer = -2
        embed_reduce_type = "mean"
    elif configs[1] == 'glove':
        embed_model = "glove.6B.300d"
        embed_format = "pickle"
        embed_layer = 0
        embed_reduce_type = "mean"
    elif configs[1] == 'fast':
        embed_model = "fasttext"
        embed_format = "json"
        embed_layer = 0
        embed_reduce_type = "mean"
    elif configs[1] == "turing":
        embed_model = "xlm-roberta-base"
        embed_format = "pickle"
        embed_layer = -2
        embed_reduce_type = "mean"
    else:
        raise NotImplementedError(f"Data config for {configs[1]} not yet implemented.")

    allow_multiple_values = True
    consider_grouping_operations = True
    for config in configs[3:]:
        if config == 'single':
            allow_multiple_values = False
            consider_grouping_operations = False
        elif config == 'x_group':
            consider_grouping_operations = False
        else:
            raise NotImplementedError(f"Data config for {config} not yet implemented.")

    search_types = convert_ana_types(search_types_str)
    previous_types = convert_ana_types(previous_types_str)
    input_types = convert_ana_types(input_types_str)

    train_ratio = 0.7
    valid_ratio = 0.1
    test_ratio = 0.2

    return DataConfig(corpus_path=corpus_path, unified_ana_token=unified_ana_token,
                      use_field_type=use_field_type, use_field_role=use_field_role, allow_agg_func=allow_agg_func,
                      use_binary_tags=use_binary_tags,
                      use_data_features=use_data_features, use_semantic_embeds=use_semantic_embeds,
                      allow_multiple_values=allow_multiple_values,
                      consider_grouping_operations=consider_grouping_operations,
                      field_permutation=field_permutation,
                      embed_model=embed_model, embed_format=embed_format,
                      embed_layer=embed_layer, embed_reduce_type=embed_reduce_type,
                      num_train_analysis=num_train_analysis,
                      search_types=search_types, searching_all_type=search_types_str.startswith('all'),
                      input_types=input_types,
                      model_types=previous_types, lang=lang,
                      train_ratio=train_ratio, valid_ratio=valid_ratio, test_ratio=test_ratio,
                      empirical_study=empirical_study, limit_search_group=limit_search_group)
