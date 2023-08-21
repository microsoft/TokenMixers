# --------------------------------------------------------
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License
# Written by Zhipeng Huang
# --------------------------------------------------------
import torch
from torch import nn
from typing import Optional

from .activation import build_activation_layer


def get_complex_activation_fn(
        act_type: Optional[str] = "relu",
        num_parameters: Optional[int] = -1,
        inplace: Optional[bool] = True,
        negative_slope: Optional[float] = 0.1,
        *args,
        **kwargs
) -> nn.Module:
    """
    Helper function to get activation (or non-linear) function
    """
    class ComplexAct(nn.Module):
        '''
        Naive approach to complex batch norm, perform batch norm independently on real and imaginary part.
        '''
        def __init__(self, act_type, num_parameters, negative_slope, inplace, *args, **kwargs):
            super(ComplexAct, self).__init__()
            self.act_r = build_activation_layer(
                act_type=act_type,
                num_parameters=num_parameters,
                negative_slope=negative_slope,
                inplace=inplace,
                *args,
                **kwargs
            )
            self.act_i = build_activation_layer(
                act_type=act_type,
                num_parameters=num_parameters,
                negative_slope=negative_slope,
                inplace=inplace,
                *args,
                **kwargs
            )

        def forward(self, input):
            return self.act_r(input.real).type(torch.complex64) + 1j * self.act_i(input.imag).type(torch.complex64)

    return ComplexAct(
        act_type=act_type,
        num_parameters=num_parameters,
        negative_slope=negative_slope,
        inplace=inplace,
        *args,
        **kwargs
    )

def get_activation_fn(
    act_type: Optional[str] = "relu",
    num_parameters: Optional[int] = -1,
    inplace: Optional[bool] = True,
    negative_slope: Optional[float] = 0.1,
    *args,
    **kwargs
) -> nn.Module:
    """
    Helper function to get activation (or non-linear) function
    """
    return build_activation_layer(
        act_type=act_type,
        num_parameters=num_parameters,
        negative_slope=negative_slope,
        inplace=inplace,
        *args,
        **kwargs
    )
