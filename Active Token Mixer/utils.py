# --------------------------------------------------------
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License
# Written by Guoqiang Wei
# --------------------------------------------------------

import io
import os
import sys
import time
import logging
import functools
import subprocess

from termcolor import colored

from collections import Mapping, OrderedDict

from collections import defaultdict

from timm.utils import get_state_dict

import torch.distributed as dist
from collections import Counter
import typing
from typing import Any, List

from fvcore.nn import flop_count
from fvcore.nn.jit_handles import get_shape, conv_flop_count

import torch
from torch.utils.cpp_extension import CUDA_HOME


def atm_flops_jit(inputs: List[Any], outputs: List[Any]) -> typing.Counter[str]:
    x, w = inputs[:2]
    x_shape, w_shape, out_shape = (get_shape(x), get_shape(w), get_shape(outputs[0]))
    assert w_shape[-1] == 1 and w_shape[-2] == 1, w_shape

    return Counter({"conv": conv_flop_count(x_shape, w_shape, out_shape)})


def load_checkpoint_for_ema(model_ema, checkpoint):
    mem_file = io.BytesIO()
    torch.save({"state_dict_ema": checkpoint}, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)

    
@functools.lru_cache()
def create_logger(output_dir, dist_rank=0):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmt = '[%(asctime)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def collect_sys_info() -> dict:
    """
    collect system information
    """
    sys_info = {}
    sys_info['Python'] = sys.version.replace('\n', '')

    cuda_available = torch.cuda.is_available()
    sys_info['CUDA available'] = cuda_available

    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            sys_info['GPU ' + ','.join(device_ids)] = name

        sys_info['CUDA_HOME'] = CUDA_HOME

        if CUDA_HOME is not None and os.path.isdir(CUDA_HOME):
            try:
                nvcc = os.path.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(
                    f'"{nvcc}" -V | tail -n1', shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            sys_info['NVCC'] = nvcc

    try:
        gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
        gcc = gcc.decode('utf-8').strip()
        sys_info['GCC'] = gcc
    except subprocess.CalledProcessError:
        sys_info['GCC'] = 'n/a'

    # pytorch
    sys_info['PyTorch'] = torch.__version__

    # torchvision
    try:
        import torchvision
        sys_info['TorchVision'] = torchvision.__version__
    except ModuleNotFoundError:
        pass

    # timm
    try:
        import timm
        sys_info['Timm'] = timm.__version__
    except ModuleNotFoundError:
        pass

    return sys_info


def dict_to_string(data, sort_keys: bool = False) -> str:
    """
    convert dict to string for elegent printing style
    """
    if not isinstance(data, Mapping):
        return f"the input is not a dict, but got {type(data)}"
    else:
        _str = ""
        _max = max([len(k) for k in data.keys()])
        if sort_keys:
            data = OrderedDict(sorted(data.items()))
        for k, v in data.items():
            _str += f"{k:{_max}}:    {v}\n"
        return _str[:-1]


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def save_checkpoint(args, save_path, epoch, model, model_ema, loss_scaler, acc_max, optimizer, lr_scheduler, logger):
    save_state = {
        'model': model.state_dict(),
        'model_ema': get_state_dict(model_ema),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
        'acc_max': acc_max,
        'epoch': epoch,
        'args': args
    }

    logger.info(f">>> saving {save_path}")
    torch.save(save_state, save_path)


def cal_flops(model):
    model_mode = model.training
    model.eval()
    pseudo_input = torch.rand(1, 3, 224, 224)
    flops_dict, *_ = flop_count(model, pseudo_input, supported_ops={"torchvision::deform_conv2d": atm_flops_jit})
    flops_count = sum(flops_dict.values())
    model.train(model_mode)

    return flops_count


def throughput(model, data_loader):
    with torch.no_grad():
        model.eval()
        for idx, (images, _) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            batch_size = images.shape[0]
            for i in range(30):
                model(images)
            torch.cuda.synchronize()
            if dist.get_rank() == 0:
                print(f">>> throughput averaged over 30 times")
            tic0 = time.time()
            for i in range(30):
                model(images)
            torch.cuda.synchronize()
            tic1 = time.time()
            if dist.get_rank() == 0:
                print(f"     batch_size: {batch_size} | throughput: {30 * batch_size / (tic1 - tic0)}")
            return
