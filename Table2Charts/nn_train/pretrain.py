# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import logging
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data import Index, QValueDataset, DEFAULT_LANGUAGES, DEFAULT_FEATURE_CHOICES, DEFAULT_ANALYSIS_TYPES
from helper import construct_data_config, create_model, prepare_model, save_ddp_checkpoint
from model import DEFAULT_MODEL_SIZES, DEFAULT_MODEL_NAMES
from os import path, getpid
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from util import num_params, get_num_params

from .trainer import Trainer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def get_arguments():
    parser = argparse.ArgumentParser()

    # Model loading configurations
    parser.add_argument('-p', '--pre_model_file', type=str, metavar='PATH',
                        help='file path to a previously trained model (as the starting point)')

    # Model choose configurations
    parser.add_argument("--model_name", choices=DEFAULT_MODEL_NAMES, default="cp", type=str)

    # Data and BERT configurations
    parser.add_argument("--corpus_path", type=str, required=True, help="The corpus path for metadata task.")
    parser.add_argument("--lang", "--specify_lang", choices=DEFAULT_LANGUAGES, default='en', type=str,
                        help="Specify the header language(s) to load tables.")
    parser.add_argument("--model_size", choices=DEFAULT_MODEL_SIZES, required=True, type=str)
    parser.add_argument('-m', "--model_save_path", default="/storage/models/", type=str)
    parser.add_argument('--features', choices=DEFAULT_FEATURE_CHOICES, default="all-mul_bert", type=str,
                        help="Limit the data loading and control the feature ablation.")
    parser.add_argument('-s', '--search_type', choices=DEFAULT_ANALYSIS_TYPES, type=str, required=True,
                        help="Determine which data to load and what types of analysis to search.")
    parser.add_argument('--input_type', choices=DEFAULT_ANALYSIS_TYPES, type=str, default=None,
                        help="Determine which data to load. This parameter is prior to --search_type.")
    parser.add_argument('--previous_type', choices=DEFAULT_ANALYSIS_TYPES, type=str,
                        help="Tell the action space information of pre_model_file/model_file."
                             "Bar grouping should be the same as in data_constraint.")
    parser.add_argument('--field_permutation', default=False, dest='field_permutation', action='store_true',
                        help="Whether to randomly permutate table fields when training.")
    parser.add_argument('--unified_ana_token', default=False, dest='unified_ana_token', action='store_true',
                        help="Whether to use unified analysis token [ANA] instead of concrete type tokens.")

    # Training configurations
    parser.add_argument("--epochs", default=30, type=int, help="Number of epoch for pretrain.")
    parser.add_argument('--restart_epoch', default=-1, type=int, metavar='N',
                        help='if the pretrain model is from an interrupted model saved by this script, '
                             'reload and restart from next epoch')
    parser.add_argument('--nprocs', default=torch.cuda.device_count(), type=int, metavar='N',
                        help='number of processes, one for each GPU.')
    parser.add_argument('--apex', default=False, dest='apex', action='store_true',
                        help="Use NVIDIA Apex DistributedDataParallel instead of the PyTorch one.")
    parser.add_argument("--num_workers", default=2, type=int, help="Number of workers (per GPU) for data loading.")
    parser.add_argument("--negative_weight", default=0.2, type=float, help="Negative class weight for NLLLoss.")
    # TODO: try to increase batch size to get faster training speed (while not OOM)
    parser.add_argument("--train_batch_size", default=64, type=int, help="Batch size (per GPU) for training.")
    parser.add_argument("--valid_batch_size", default=64, type=int, help="Batch size (per GPU) for validating.")
    parser.add_argument("--freeze_embed", default=False, dest='freeze_embed', action='store_true',
                        help="Whether to freeze params in embedding layer."
                             "Only take effect when --pre_model_file is available.")
    parser.add_argument("--freeze_encoder", default=False, dest='freeze_encoder', action='store_true',
                        help="Whether to freeze params in encoder layers."
                             "Only take effect when --pre_model_file is available.")
    parser.add_argument("--fresh_decoder", default=False, dest='fresh_decoder', action='store_true',
                        help="Whether to re-initialize params in decoding layers (attention layer, copy layer, etc.)"
                             "Only take effect when --pre_model_file is available.")
    parser.add_argument("--num_train_analysis", type=int,
                        help="Number of Analysis each ana_type for training.")

    # Other configurations
    parser.add_argument("--log_freq", type=int, default=2000, help="frequency of logging")
    parser.add_argument("--summary_path", default="/storage/summaries/", type=str, help='tensorboard summary path')

    return parser.parse_args()


# TODO: Pass new cUids here
def prepare_data(tUIDs: list, is_train: bool, args):
    data_config = construct_data_config(args, is_train)

    q_values = QValueDataset(tUIDs, data_config, is_train=is_train)
    dist_sampler = DistributedSampler(q_values)  # We assume dist is available here, otherwise set num_replica and rank!
    batch_size = args.train_batch_size if is_train else args.valid_batch_size
    data_loader = DataLoader(q_values, batch_size=batch_size, sampler=dist_sampler, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, collate_fn=q_values.collate)

    return q_values, data_loader, dist_sampler


def train_parallel(rank, args):
    """
    Process using ParallelAgent. Limit: all agent returns QValues that feed into a unique DQN.
    :param rank: process index
    :param args: parsed arguments
    """
    logger = logging.getLogger(f"train_parallel({rank}@{getpid()})")
    logger.info("pretrain() started!")
    device = torch.device("cuda:{}".format(rank) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend=dist.Backend.NCCL, world_size=args.nprocs, rank=rank)

    # Initialize and prepare model
    logger.info(f"Constructing {args.model_name} model...")
    model, experiment_name = create_model(args)
    if rank == 0:
        logger.info(f"{args.model_name} #parameters = {num_params(model)}")
        for pair in get_num_params(model).items():
            logger.info(f"{pair[0]}: {pair[1]}")
        logger.info(f"Total embed params: {sum(num_params(sub_model) for sub_model in model.get_embed_modules())}")
        logger.info(f"Total encoder params: {sum(num_params(sub_model) for sub_model in model.get_encoder_modules())}")
        logger.info(f"Total decoder params: {sum(num_params(sub_model) for sub_model in model.get_decoder_modules())}")

    # Data loading
    # TODO: index in memory with data on disk?
    # TODO: is it possible to load dataset once and share across processes?
    data_config = construct_data_config(args)
    logger.info("Loading index...")
    index = Index(data_config)
    logger.info("Loading training data...")
    train_dataset, train_dataloader, train_sampler = prepare_data(index.train_tUIDs(), True, args)
    logger.info(f"{len(train_dataset)} training samples in {len(train_dataloader)} batches.")
    logger.info("Loading validation data...")
    valid_dataset, valid_dataloader, _ = prepare_data(index.valid_tUIDs(), False, args)
    logger.info(f"{len(valid_dataset)} validation samples in {len(valid_dataloader)} batches.")

    # for param in model.named_parameters():
    #     logger.info(f"{param[0]} : {param[1].size()}")
    ddp, optimizer, criterion = prepare_model(model, device, args)
    save_dir = path.join(args.model_save_path, experiment_name)

    logger.info(f"Start pretraining on {device}.")
    writer = SummaryWriter(log_dir=path.join(args.summary_path, experiment_name))
    trainer = Trainer(ddp, device, rank, args.apex, train_dataloader, valid_dataloader, train_sampler.set_epoch,
                      optimizer, criterion, writer, args.log_freq)
    for epoch in range(args.restart_epoch + 1, args.epochs):
        logger.info("Starting EP %d" % epoch)
        trainer.train(epoch)

        # Save model
        if rank == 0:  # Save model checkpoint
            output_path = save_ddp_checkpoint(save_dir, epoch, trainer.model, trainer.optim)
            logger.info("EP%d Model Saved on: " % epoch + output_path)

        if valid_dataloader is not None:
            logger.info("Start validation of EP%d" % epoch)
            trainer.test(epoch)
    logger.info("pretrain() finished!")


def pretrain(args):
    args.mode = None

    logger = logging.getLogger("pretrain()")
    logger.info(f"Pretrain Args: {args}")

    data_config = construct_data_config(args)
    logger.info("DataConfig: {}".format(vars(data_config)))

    mp.spawn(train_parallel, args=(args,), nprocs=args.nprocs)


if __name__ == "__main__":
    pretrain(get_arguments())
