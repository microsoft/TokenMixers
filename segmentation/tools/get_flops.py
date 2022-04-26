import argparse
import typing
from typing import Any, List
from collections import Counter

import torch

from mmcv import Config
from mmseg.models import build_segmentor

from fvcore.nn.jit_handles import get_shape, conv_flop_count
from fvcore.nn import flop_count

import activemlp


def atm_flops_jit(inputs: List[Any], outputs: List[Any]) -> typing.Counter[str]:
    x, w = inputs[:2]
    x_shape, w_shape, out_shape = (get_shape(x), get_shape(w), get_shape(outputs[0]))
    assert w_shape[-1] == 1 and w_shape[-2] == 1, w_shape

    return Counter({"conv": conv_flop_count(x_shape, w_shape, out_shape)})


def main(args):
    input_shape = args.shape
    assert len(args.shape) == 2, f"expected shape with [H, W], but got {args.shape}"

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.format(model.__class__.__name__)
        )

    inp = torch.rand((1, 3) + tuple(input_shape)).cuda()

    print(f">>> model: {cfg.model.backbone.type} from with: {args.config} | shape={input_shape}")
    flops_dict, *_ = flop_count(model, inp, supported_ops={"torchvision::deform_conv2d": atm_flops_jit})
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    count = sum(flops_dict.values())
    print(f"    FLOPs: {count:.3f}G\n"
          f"    #param: {n_parameters / 1e6:.3f}M")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('shape',    type=int, nargs='+', help='input image shape with [H, W]')
    parser.add_argument('--config', type=str, help='path to model config')
    args = parser.parse_args()

    main(args)

