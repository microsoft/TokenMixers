# --------------------------------------------------------
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License
# Written by Zhipeng Huang
# --------------------------------------------------------

from torch import nn, Tensor
from typing import Tuple

from . import register_act_fn


@register_act_fn(name="sigmoid")
class Sigmoid(nn.Sigmoid):
    """
    Applies the sigmoid function
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0
