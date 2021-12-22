import argparse
import json
import logging
import os
import queue
from threading import Thread
from time import perf_counter
from typing import List

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from data import DEFAULT_LANGUAGES, DEFAULT_FEATURE_CHOICES, DEFAULT_ANALYSIS_TYPES, get_data_config, DataConfig, \
    SpecialTokens, QValue, Index, AnaType
from model import CopyNet, DEFAULT_MODEL_SIZES, DEFAULT_MODEL_NAMES, get_cp_config
from search import merge_eval_info
from search.agent import DEFAULT_SEARCH_LIMITS, get_search_config, SearchConfig, \
    Agent, ParallelAgents, BeamDrillDownAgent
from util import load_checkpoint, to_device, time_str

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

AGENT_NAMES = sorted(["drill_down"])


def parse_args():
    parser = argparse.ArgumentParser(description="Concurrent Test Search Agents")

    parser.add_argument("--empirical_study", default=False, dest='empirical_study', action='store_true',
                        help='Given table json, then get small-scale recommendation results.')
    parser.add_argument("--empirical_corpus_path", default="/home/exp/pivot-clean-data/", type=str)
    parser.add_argument("--empirical_log_path", default="/home/exp/empirical_result", type=str)
    parser.add_argument("--corpus_path", type=str, required=True, help="The corpus path for metadata task.")
    parser.add_argument("--lang", "--specify_lang", choices=DEFAULT_LANGUAGES, default='en', type=str,
                        help="Specify the header language(s) to load tables.")

    # Model choose configurations
    parser.add_argument("--model_name", choices=DEFAULT_MODEL_NAMES, default="cp", type=str)

    # Model and evaluation paths
    parser.add_argument('-m', '--model_dir_path', default="/storage/models/random/", type=str, metavar='PATH',
                        help='dir path holding the models (as the starting point)')
    parser.add_argument('-f', '--model_file', type=str, metavar='FILENAME')
    parser.add_argument('-l', '--log_save_path', default="evaluations/test", type=str, metavar='PATH',
                        help='subdir path of model_dir_path to log the evaluation metrics during testing')

    # Experiment settings
    parser.add_argument('--model_size', choices=DEFAULT_MODEL_SIZES, required=True, type=str)
    parser.add_argument('--features', choices=DEFAULT_FEATURE_CHOICES, default="all-mul_bert", type=str,
                        help="Limit the data loading and control the feature ablation.")
    parser.add_argument('-s', '--search_type', choices=DEFAULT_ANALYSIS_TYPES, type=str, required=True,
                        help="Determine which data to load and what types of analysis to search.")
    parser.add_argument('--test_type', choices=DEFAULT_ANALYSIS_TYPES, type=str, default=None,
                        help="Determine which chart type to search. This parameter is prior to --search_type.")
    parser.add_argument('--input_type', choices=DEFAULT_ANALYSIS_TYPES, type=str, default=None,
                        help="Determine which data to load. This parameter is prior to --search_type.")
    parser.add_argument('--previous_type', choices=DEFAULT_ANALYSIS_TYPES, type=str,
                        help="Tell the action space information of pre_model_file/model_file."
                             "Bar grouping should be the same as in --features.")
    parser.add_argument('--limit_search_group', action='store_true',
                        help="If it's True, not search Group.(We don't have Group in Plotly).")
    parser.add_argument('--mode', type=str, choices=['FULL', 'EMB'],
                        help='use full features models or only embedding features models')
    parser.add_argument('--log_freq_agent', default=100, type=int, metavar='N',
                        help='number of tables to print agent evaluation metrics')
    parser.add_argument('--use_valid_set', action='store_true',
                        help='set this flag if validation set should be used instead of test set.')
    parser.add_argument('--test_whole_plotly', action='store_true',
                        help='set this flag if test whole Plotly dataset.')
    parser.add_argument('--web_table', action='store_true',
                        help='set this flag if test web table dataset.')
    parser.add_argument('--unified_ana_token', default=False, dest='unified_ana_token', action='store_true',
                        help="Whether to use unified analysis token [ANA] instead of concrete type tokens.")

    # Parallel search settings (nagents * beam_size is the max batch size)
    parser.add_argument('--search_algo', metavar='ALGORITHM', default='drill_down', choices=AGENT_NAMES,
                        help='search agent algorithm: ' + ' | '.join(AGENT_NAMES) + ' (default: drill_down)')
    parser.add_argument('--nprocs', default=24, type=int, metavar='N',
                        help='number of concurrent searching processes')
    parser.add_argument('--nagents', default=128, type=int, metavar='N',
                        help='number of agents (tables) to handle in parallel')
    parser.add_argument('--nthreads', default=5, type=int, metavar='N',
                        help='number of threads to handle parallel agents')

    # Drill agent parameters
    parser.add_argument('--search_limits', choices=DEFAULT_SEARCH_LIMITS, default="e200-b8-r4c2", type=str,
                        help="Search config option")
    # Staged agent parameters
    parser.add_argument('--add_sep_to_dim', action='store_true',
                        help='set this flag if dim combinations without SEP should be adopted')
    parser.add_argument('--force_keep_empty_dim', action='store_true',
                        help='set this flag if we will always keep the empty dim combination')

    # Task parameters
    parser.add_argument('--test_field_selections', action='store_true',
                        help='set this flag if we only want to test field selection')
    parser.add_argument('--test_design_choices', action='store_true',
                        help='set this flag if we only want to test charting (visualization)')

    return parser.parse_args()


def construct_data_config(args) -> DataConfig:
    data_config = get_data_config(args.corpus_path, args.features,
                                  args.search_type, args.previous_type, args.input_type,
                                  args.unified_ana_token, None, False, lang=args.lang,
                                  empirical_study=args.empirical_study, mode=args.mode,
                                  limit_search_group=args.limit_search_group)
    if args.model_name == "cp":
        data_config.need_field_indices = True
    if args.empirical_study:
        data_config.empirical_study = True
        data_config.corpus_path = args.empirical_corpus_path

    return data_config


def construct_search_config(args, data_config) -> SearchConfig:
    load_ground_truth = False if args.web_table else True
    search_config = get_search_config(load_ground_truth, args.search_limits,
                                      search_all_types=data_config.search_all_types,
                                      search_single_type=AnaType.from_raw_str(args.test_type)
                                      if args.test_type is not None else None,
                                      log_path=args.empirical_log_path if args.empirical_study or args.web_table else None,
                                      test_field_selections=args.test_field_selections,
                                      test_design_choices=args.test_design_choices)
    # TODO: args.log_save_path currently not used

    return search_config


def create_model(device, args):
    logger = logging.getLogger("model config()")
    data_config = construct_data_config(args)

    if args.model_name == "cp":
        cp_config = get_cp_config(data_config, args.model_size)
        model = CopyNet(cp_config)
        logger.info("CopyNetConfig: {}".format(vars(cp_config)))
    else:
        raise NotImplementedError(f"{args.model_name} not yet implemented.")

    if args.model_file:
        load_checkpoint(os.path.join(args.model_dir_path, args.model_file), model, device=device)
    model.to(device)
    model.eval()
    return model


def create_agent(tUID: str, special_tokens: SpecialTokens, args) -> Agent:
    data_config = construct_data_config(args)
    search_config = construct_search_config(args, data_config)

    if args.search_algo == "drill_down":
        return BeamDrillDownAgent(tUID, data_config, special_tokens, search_config)
    else:
        raise ValueError(f"No {args.search_algo} search agent not implemented!")


def feed_batch_nn(samples: List[QValue], model: torch.nn.Module, device: str, config: DataConfig) -> np.ndarray:
    if len(samples) == 0:
        return np.empty([0])
    with torch.no_grad():
        data = QValue.collate(samples, config, False)
        del data["values"]  # Not useful in evaluation
        data = to_device(data, device)
        return model(data["state"], data["actions"]).detach()[:, :, 1].exp().cpu()


def process_parallel(index, device_count, special_tokens: SpecialTokens,
                     tuid_queue: mp.Queue, info_queue: mp.Queue, args):
    """
    Process using ParallelAgent. Limit: all agent returns QValues that feed into a unique DQN.
    :param index: process index
    :param device_count: number of GPUs
    :param special_tokens: special tokens
    :param tuid_queue: consume this queue
    :param info_queue: put evaluation results into this queue
    :param args: parsed arguments
    """
    logger = logging.getLogger(f"process_parallel ({index}@{os.getpid()})")
    start_time = perf_counter()
    device = index % device_count
    torch.cuda.set_device(device)
    dqn = create_model(device, args)

    logger.info(f"Start to search on {device}.")
    data_config = construct_data_config(args)

    agents = ParallelAgents(max_workers=args.nthreads)
    while True:
        # Add more agents.
        while agents.remaining() < args.nagents and not tuid_queue.empty():
            try:
                tUID = tuid_queue.get(block=False)
                agents.add(create_agent(tUID, special_tokens, args))
            except queue.Empty:
                continue
            except ValueError:  # No user created PivotTables or Source too large
                continue
        if agents.remaining() == 0:
            try:
                tUID = tuid_queue.get(timeout=30)
                agents.add(create_agent(tUID, special_tokens, args))
            except queue.Empty:
                info_queue.put_nowait(None)  # Finish!
                logger.info(f"Finish {agents.finished()}. {perf_counter() - start_time:.1f}s passed.")
                agents.shutdown()
                return
            except ValueError:
                continue

        futures = agents.step([lambda samples: feed_batch_nn(samples, dqn, device, data_config)])  # Work on only 1 GPU
        finished_info = agents.update([(lambda: future.result()) for future in futures])
        for info in finished_info:
            info_queue.put_nowait(info)


def summary(info_queue: mp.Queue, args):
    logger = logging.getLogger(f"summary(@{os.getpid()})")
    start_time = perf_counter()
    logger.info("Start to summarize info.")

    info_list = []
    finished_proc = 0
    log_file_path = os.path.join(args.log_save_path, f"[test-summary]{time_str()}.log")
    with open(log_file_path, "w") as log_file:
        while True:
            info = info_queue.get()
            if info is not None:
                info_list.append(info)
            else:
                finished_proc += 1
                if finished_proc == args.nprocs:
                    break
                else:
                    continue
            if len(info_list) % args.log_freq_agent == 0:
                logger.info(f"{len(info_list)}th done. {perf_counter() - start_time:.1f}s passed.")
                log_file.write("[from {} to {}]\n".format(len(info_list) - args.log_freq_agent + 1, len(info_list)))
                log_file.write(json.dumps(merge_eval_info(info_list[-args.log_freq_agent:]), sort_keys=True, indent=4))
                log_file.write("\n\n[{} only]\n".format(len(info_list)))
                log_file.write(json.dumps(info_list[-1], sort_keys=True, indent=4))
                log_file.write("\n\n")
                log_file.flush()

        log_file.write("[all {}]\n".format(len(info_list)))
        summary_info = merge_eval_info(info_list)
        log_file.write(json.dumps(summary_info, sort_keys=True, indent=4))
        log_file.flush()
        logger.info(f"Finish summarize {len(info_list)} info into {log_file_path}. "
                    f"{perf_counter() - start_time:.1f}s passed.")
        if "evaluation" in summary_info:
            complete_info = summary_info["evaluation"]["stages"]["complete"]
            logger.info(f"Complete recall info: {complete_info}")


def test(args):
    logger = logging.getLogger("test()")
    logger.info(f"Test Args: {args}")

    data_config = construct_data_config(args)
    special_tokens = SpecialTokens(data_config)
    logger.info("DataConfig: {}".format(vars(data_config)))

    if args.model_name == "cp":
        cp_config = get_cp_config(data_config, args.model_size)
        logger.info("CPConfig: {}".format(vars(cp_config)))
    else:
        raise NotImplementedError(f"{args.model_name} not yet implemented.")

    if args.log_save_path is not None:
        args.log_save_path = os.path.join(args.model_dir_path, args.log_save_path)
        os.makedirs(args.log_save_path, exist_ok=True)

    search_config = construct_search_config(args, data_config)
    logger.info("SearchConfig: {}".format(vars(search_config)))

    logger.info("Loading index...")
    index = Index(data_config)

    device_count = torch.cuda.device_count()
    if args.use_valid_set:
        tUIDs = index.valid_tUIDs()
    elif args.test_whole_plotly or args.web_table or args.empirical_study:
        tUIDs = index.get_tUIDs()
    else:
        tUIDs = index.test_tUIDs()

    logger.info(f"Testing {len(tUIDs)} files from {index.config.index_path()}")
    if args.nprocs > 1 and device_count:  # Will start args.nprocs processes, evenly distributed on all GPUs.
        context = mp.get_context(method='spawn')
        cq = context.Queue()
        iq = context.Queue()
        for tUID in tUIDs:
            cq.put(tUID)
        thread = Thread(target=summary, args=(iq, args))
        thread.start()
        if args.search_algo == "drill_down":
            mp.spawn(process_parallel, (device_count, special_tokens, cq, iq, args), args.nprocs)
        thread.join()
    else:
        cq = mp.Queue()
        iq = mp.Queue()
        for tUID in tqdm(tUIDs):
            cq.put(tUID)
        thread = Thread(target=summary, args=(iq, args))
        thread.start()
        if args.search_algo == "drill_down":
            process_parallel(0, device_count, special_tokens, cq, iq, args)
        thread.join()
    logger.info("Test finished!")


if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn', force=True)
    test(parse_args())
