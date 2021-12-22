from typing import Optional
import numpy as np
from data import AnaType


class SearchConfig:
    def __init__(self, load_ground_truth: bool, search_all_types: bool = False,
                 dim_count: int = 600, com_count: int = 1100, max_rc: tuple = (4, 2),
                 expand_limit: int = 200, time_limit: float = np.inf,
                 frontier_size: int = 300, beam_size: int = 8, min_threshold: float = 0.,
                 log_path: Optional[str] = None, search_single_type: AnaType = None,
                 test_field_selections: bool = False, test_design_choices: bool = False):
        """
        :param load_ground_truth:
        :param dim_count:  max number of dim(fields)
        :param com_count:  max number of steps for com task
        :param max_rc:
        :param expand_limit:
        :param time_limit:
        :param frontier_size:
        :param beam_size:
        :param min_threshold:
        :param log_path:
        :param search_single_type: limit searching to the specified ana_type.
        If it is None, this limitation will not be applied.
        :param test_field_selections: whether to only test field selection
        :param test_design_choices: whether to only test charting (visualization)
        """
        self.load_ground_truth = load_ground_truth
        self.search_all_types = search_all_types
        self.search_single_type = search_single_type

        self.test_field_selections = test_field_selections
        self.test_design_choices = test_design_choices

        self.max_rc = max_rc
        self.expand_limit = expand_limit
        self.time_limit = time_limit
        self.dim_count = dim_count
        self.com_count = com_count

        self.frontier_size = frontier_size
        self.beam_size = beam_size
        self.min_threshold = min_threshold

        self.log_path = log_path


DEFAULT_SEARCH_LIMITS = ["e200-b4-r4c2", "e200-b8-r4c2", "e200-b4-na", "e200-b8-na",
                         "e100-b4-r4c2", "e100-b8-r4c2", "e100-b4-na", "e100-b8-na",
                         "e50-b4-r4c2", "e50-b4-na"]


def get_ebrc_config(load_ground_truth: bool,
                    test_field_selections: bool = False,
                    test_design_choices: bool = False,
                    expand_limit: int = 200, beam_size: int = 8,
                    r_limit: int = 1000, c_limit: int = 1000,
                    log_path: Optional[str] = None):
    return SearchConfig(load_ground_truth, expand_limit=expand_limit, beam_size=beam_size,
                        max_rc=(r_limit, c_limit), log_path=log_path,
                        test_field_selections=test_field_selections,
                        test_design_choices=test_design_choices)


def get_search_config(load_ground_truth: bool, limits: str,
                      search_all_types: bool = False, log_path: Optional[str] = None,
                      search_single_type: AnaType = None,
                      test_field_selections: bool = False,
                      test_design_choices: bool = False) -> SearchConfig:
    if limits == "e200-b4-r4c2":
        config = get_ebrc_config(load_ground_truth, test_field_selections, test_design_choices, beam_size=4, r_limit=4, c_limit=2)
    elif limits == "e200-b8-r4c2":
        config = get_ebrc_config(load_ground_truth, test_field_selections, test_design_choices, r_limit=4, c_limit=2)
    elif limits == "e200-b4-na":
        config = get_ebrc_config(load_ground_truth, test_field_selections, test_design_choices, beam_size=4)
    elif limits == "e200-b8-na":
        config = get_ebrc_config(load_ground_truth, test_field_selections, test_design_choices)
    elif limits == "e100-b4-r4c2":
        config = get_ebrc_config(load_ground_truth, test_field_selections, test_design_choices, expand_limit=100, beam_size=4, r_limit=4, c_limit=2)
    elif limits == "e100-b8-r4c2":
        config = get_ebrc_config(load_ground_truth, test_field_selections, test_design_choices, expand_limit=100, r_limit=4, c_limit=2)
    elif limits == "e100-b4-na":
        config = get_ebrc_config(load_ground_truth, test_field_selections, test_design_choices, expand_limit=100, beam_size=4)
    elif limits == "e100-b8-na":
        config = get_ebrc_config(load_ground_truth, test_field_selections, test_design_choices, expand_limit=100)
    elif limits == "e50-b4-r4c2":
        config = get_ebrc_config(load_ground_truth, test_field_selections, test_design_choices, expand_limit=50, beam_size=4, r_limit=4, c_limit=2)
    elif limits == "e50-b4-na":
        config = get_ebrc_config(load_ground_truth, test_field_selections, test_design_choices, expand_limit=50, beam_size=4)
    else:
        raise NotImplementedError(f"Agent config for {limits} not yet implemented.")

    # True : search for all analysis/chart types
    # False: only search for table available analysis/chart type(s)
    config.search_all_types = search_all_types
    config.search_single_type = search_single_type

    config.log_path = log_path
    return config
