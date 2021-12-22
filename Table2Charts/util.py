import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer


def time_str():
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M")


def to_device(data, device, non_blocking=False):
    if isinstance(data, torch.Tensor):
        return data.to(device=device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        return {k: to_device(value, device, non_blocking) for k, value in data.items()}
    else:
        return data


def data_shapes(data):
    if isinstance(data, torch.Tensor):
        return data.size()
    elif isinstance(data, dict):
        return {k: data_shapes(value) for k, value in data.items()}
    else:
        return data


def unsqueeze(data):
    if isinstance(data, torch.Tensor):
        return data.unsqueeze(0)
    elif isinstance(data, dict):
        return {k: unsqueeze(value) for k, value in data.items()}
    else:
        raise ValueError("Could not handle!")


def scores_from_confusion(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


# Set requires_grad property for all parameters in a Module.
def set_no_calculate_gradient(module: Module):
    for param in module.parameters():
        param.requires_grad = False


def set_calculate_gradient(module: Module):
    for param in module.parameters():
        param.requires_grad = True


def save_states(epoch: int, states: dict, dir_path: str):
    """
    Saving the current states in dir_path
    See https://pytorch.org/tutorials/beginner/saving_loading_models.html
    :param epoch: current epoch number
    :param states: states dict to save
    :param dir_path: model output path
    :return: final_output_path
    """
    os.makedirs(dir_path, exist_ok=True)
    output_path = os.path.join(dir_path, "states_ep%d.pt" % epoch)
    torch.save(states, output_path)
    return output_path


def save_checkpoint(dir_path: str, epoch: int, module: Module,
                    optimizer: Optimizer = None, extra: Optional[dict] = None):
    states = {
        "epoch": epoch,
        "module_state": module.state_dict(),
    }
    if optimizer is not None:
        states["optim_state"] = optimizer.state_dict()
    if extra is not None:
        states["extra"] = extra

    return save_states(epoch, states, dir_path)


def load_states(file_path: str):
    """
    See https://pytorch.org/tutorials/beginner/saving_loading_models.html
    to learn more about cross-device loading!
    We force load checkpoint on CPU to avoid GPU RAM surge,
    see https://pytorch.org/docs/stable/torch.html#torch.load
    :param file_path: The .pt states saved
    :return: States dict in CPU
    """
    return torch.load(file_path, map_location="cpu")


def load_checkpoint(file_path: str, module: Module, optimizer: Optimizer = None, device="cpu"):
    states = load_states(file_path)
    module.load_state_dict(states["module_state"])
    module.to(device)
    if optimizer is not None:
        optimizer.load_state_dict(states["optim_state"])

    return states["epoch"], module, optimizer, states.get("extra")


def load_optimizer(file_path: str, optimizer: Optimizer):
    states = load_states(file_path)
    optimizer.load_state_dict(states["optim_state"])
    return optimizer


def load_extra(file_path: str):
    states = load_states(file_path)
    return states.get("extra")


def log_params(args, data_config, model_config, experiment_name):
    # log experiment arguments to summary folder
    summary_dir_path = os.path.join(args.summary_path, experiment_name)
    if not os.path.exists(summary_dir_path):
        try:
            os.makedirs(summary_dir_path)
        except:
            pass
    with open(os.path.join(summary_dir_path, "params.log"), 'w') as f:
        f.write("Experiment " + experiment_name + "\n")
        f.write("--------------------------------------------------\n")
        f.write("pretrain parameters:\n")
        for name, value in args._get_kwargs():
            f.write("{0}{1}\n".format(format(str(name), " <30"), format(str(value), " <30")))
        f.write("\ndata configs:\n")
        for name, value in data_config.__dict__.items():
            f.write("{0}{1}\n".format(format(str(name), " <30"), format(str(value), " <30")))
        f.write("\nmodel configs:\n")
        for name, value in model_config.__dict__.items():
            f.write("{0}{1}\n".format(format(str(name), " <30"), format(str(value), " <30")))


def num_params(model: Module):
    return sum([param.nelement() for param in model.parameters()])


def get_num_params(model: Module):
    param2num = dict()
    for name, param in model.named_parameters():
        param2num[name] = (param.nelement(), param.size())
    return param2num
