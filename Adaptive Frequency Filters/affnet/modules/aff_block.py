# --------------------------------------------------------
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License
# Written by Zhipeng Huang
# --------------------------------------------------------
import einops
import numpy as np
from torch import nn, Tensor
import math
import torch
from torch.nn import functional as F
from typing import Optional, Dict, Tuple, Union, Sequence

from . import InvertedResidual
from .transformer import TransformerEncoder, LinearAttnFFN
from .base_module import BaseModule
from ..misc.profiler import module_profile
from ..layers import ConvLayer, get_normalization_layer, get_activation_fn
import typing
from typing import Any, List
from einops.layers.torch import Rearrange
import math
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import time


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class AFNO2D_channelfirst(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    input shape [B N C]
    """

    def __init__(self, opts, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1,
                 hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = getattr(opts, "model.activation.sparsity_threshold", 0.01)
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        # self.norm_layer1 = get_normalization_layer(opts=opts, num_features=out_channels)
        self.act = self.build_act_layer(opts=opts)
        self.act2 = self.build_act_layer(opts=opts)

    @staticmethod
    def build_act_layer(opts) -> nn.Module:
        act_type = getattr(opts, "model.activation.name", "relu")
        neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
        inplace = getattr(opts, "model.activation.inplace", False)
        act_layer = get_activation_fn(
            act_type=act_type,
            inplace=inplace,
            negative_slope=neg_slope,
            num_parameters=1,
        )
        return act_layer

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, spatial_size=None):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape
        # x = self.fu(x)

        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        origin_ffted = x
        x = x.reshape(B, self.num_blocks, self.block_size, x.shape[2], x.shape[3])


        o1_real = self.act(
            torch.einsum('bkihw,kio->bkohw', x.real, self.w1[0]) - \
            torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[1]) + \
            self.b1[0, :, :, None, None]
        )

        o1_imag = self.act2(
            torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[0]) + \
            torch.einsum('bkihw,kio->bkohw', x.real, self.w1[1]) + \
            self.b1[1, :, :, None, None]
        )

        o2_real = (
                torch.einsum('bkihw,kio->bkohw', o1_real, self.w2[0]) - \
                torch.einsum('bkihw,kio->bkohw', o1_imag, self.w2[1]) + \
                self.b2[0, :, :, None, None]
        )

        o2_imag = (
                torch.einsum('bkihw,kio->bkohw', o1_imag, self.w2[0]) + \
                torch.einsum('bkihw,kio->bkohw', o1_real, self.w2[1]) + \
                self.b2[1, :, :, None, None]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, C, x.shape[3], x.shape[4])

        x = x * origin_ffted
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")
        x = x.type(dtype)

        return x + bias

    def profile_module(
            self, input: Tensor, *args, **kwargs
        ) -> Tuple[Tensor, float, float]:
        # TODO: to edit it
        b_sz, c, h, w = input.shape
        seq_len = h * w

        # FFT iFFT
        p_ff, m_ff = 0, 5 * b_sz * seq_len * int(math.log(seq_len)) * c
        # others
        # params = macs = sum([p.numel() for p in self.parameters()])
        params = macs = self.hidden_size * self.hidden_size_factor * self.hidden_size * 2 * 2 // self.num_blocks
        # // 2 min n become half after fft
        macs = macs * b_sz * seq_len

        # return input, params, macs
        return input, params, macs + m_ff


def remove_edge(img: np.ndarray):
    # // remove the edge of a numpy image
    return img[1:-1, 1:-1]

def save_feature(feature):
    import time
    import matplotlib.pyplot as plt
    import os
    now = time.time()
    feature = feature.detach()
    os.makedirs('visual_example', exist_ok=True)
    for i in range(feature.shape[1]):
        feature_channel = feature[0, i]
        fig, ax = plt.subplots()
        img_channel = ax.imshow(remove_edge(feature_channel.cpu().numpy()), cmap='gray')
        plt.savefig('visual_example/{now}_channel_{i}_feature.png'.format(now=str(now), i=i))
    for i in range(8):
        feature_group = torch.mean(feature[0, i * 8:(i + 1) * 8], dim=1)
        fig, ax = plt.subplots()
        img_group = ax.imshow(remove_edge(feature_group.cpu().numpy()), cmap='gray')
        plt.savefig('visual_example/{now}_group_{i}_feature.png'.format(now=str(now), i=i))

def save_kernel(origin_ffted, H, W):
    import time
    import matplotlib.pyplot as plt
    import os
    now = time.time()
    origin_ffted = origin_ffted.detach()
    kernel = torch.fft.irfft2(origin_ffted, s=(H, W), dim=(2, 3), norm="ortho")
    group_channels = kernel.shape[1] // 8
    os.makedirs('visual_example', exist_ok=True)
    for i in range(kernel.shape[1]):
        kernel_channel = kernel[0, i]
        fig, ax = plt.subplots()
        img_channel = ax.imshow(remove_edge(kernel_channel.cpu().numpy()), cmap='gray')
        plt.savefig('visual_example/{now}_channel_{i}_kernel.png'.format(now=str(now), i=i))
    for i in range(8):
        kernel_group = torch.mean(kernel[0, i*group_channels: (i+1)*group_channels], dim=0)
        fig, ax = plt.subplots()
        img_group = ax.imshow(remove_edge(kernel_group.cpu().numpy()), cmap='gray')
        plt.savefig('visual_example/{now}_group_{i}_kernel.png'.format(now=str(now), i=i))
    kernel_mean = torch.mean(kernel[0], dim=0)
    fig, ax = plt.subplots()
    img_mean = ax.imshow(remove_edge(kernel_mean.cpu().numpy()), cmap='gray')
    plt.savefig('visual_example/{now}_all_kernel.png'.format(now=str(now)))

    abs = origin_ffted.abs()
    abs_group_channels = abs.shape[1] // 8
    os.makedirs('visual_mask_example', exist_ok=True)
    for i in range(abs.shape[1]):
        abs_channel = abs[0, i]
        fig, ax = plt.subplots()
        abs_channel = ax.imshow(abs_channel.cpu().numpy(), cmap='gray')
        plt.savefig('visual_mask_example/{now}_channel_{i}_abs.png'.format(now=str(now), i=i))
    for i in range(8):
        abs_group = torch.mean(abs[0, i*abs_group_channels: (i+1)*abs_group_channels], dim=0)
        fig, ax = plt.subplots()
        img_group = ax.imshow(abs_group.cpu().numpy(), cmap='gray')
        plt.savefig('visual_mask_example/{now}_group_{i}_abs.png'.format(now=str(now), i=i))
    abs_mean = torch.mean(abs[0], dim=0)
    fig, ax = plt.subplots()
    img_mean = ax.imshow(abs_mean.cpu().numpy(), cmap='gray')
    plt.savefig('visual_mask_example/{now}_all_abs.png'.format(now=str(now)))

    real = origin_ffted.real
    real_group_channels = real.shape[1] // 8
    os.makedirs('visual_mask_example', exist_ok=True)
    for i in range(real.shape[1]):
        real_channel = real[0, i]
        fig, ax = plt.subplots()
        real_channel = ax.imshow(real_channel.cpu().numpy(), cmap='gray')
        plt.savefig('visual_mask_example/{now}_channel_{i}_real.png'.format(now=str(now), i=i))
    for i in range(8):
        real_group = torch.mean(real[0, i*real_group_channels: (i+1)*real_group_channels], dim=0)
        fig, ax = plt.subplots()
        img_group = ax.imshow(real_group.cpu().numpy(), cmap='gray')
        plt.savefig('visual_mask_example/{now}_group_{i}_mask.png'.format(now=str(now), i=i))
    real_mean = torch.mean(real[0], dim=0)
    fig, ax = plt.subplots()
    img_mean = ax.imshow(real_mean.cpu().numpy(), cmap='gray')
    plt.savefig('visual_mask_example/{now}_all_real.png'.format(now=str(now)))

    imag = origin_ffted.imag
    imag_group_channels = imag.shape[1] // 8
    os.makedirs('visual_mask_example', exist_ok=True)
    for i in range(8):
        imag_group = torch.mean(imag[0, i*imag_group_channels: (i+1)*imag_group_channels], dim=0)
        fig, ax = plt.subplots()
        img_group = ax.imshow(imag_group.cpu().numpy(), cmap='gray')
        plt.savefig('visual_mask_example/{now}_group_{i}_imag.png'.format(now=str(now), i=i))
    imag_mean = torch.mean(imag[0], dim=0)
    fig, ax = plt.subplots()
    img_mean = ax.imshow(imag_mean.cpu().numpy(), cmap='gray')
    plt.savefig('visual_mask_example/{now}_all_imag.png'.format(now=str(now)))



class Block(nn.Module):
    def __init__(self, opts, dim, hidden_size, num_blocks, double_skip, mlp_ratio=4., drop_path=0., attn_norm_layer='sync_batch_norm', enable_coreml_compatible_fn=False):
        # input shape [B C H W]
        super().__init__()
        self.norm1 = get_normalization_layer(
            opts=opts, norm_type=attn_norm_layer, num_features=dim
        )
        self.filter = AFNO2D_channelfirst(opts=opts, hidden_size=hidden_size, num_blocks=num_blocks, sparsity_threshold=0.01,
                                          hard_thresholding_fraction=1, hidden_size_factor=1) if not enable_coreml_compatible_fn else \
            AFNO2D_channelfirst_coreml(opts=opts, hidden_size=hidden_size, num_blocks=num_blocks, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = get_normalization_layer(
            opts=opts, norm_type=attn_norm_layer, num_features=dim
        )
        self.mlp = InvertedResidual(
            opts=opts,
            in_channels=dim,
            out_channels=dim,
            stride=1,
            expand_ratio=mlp_ratio,
        )
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        # x = self.filter(x)
        x = self.mlp(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        # x = self.mlp(x)
        x = self.filter(x)
        x = self.drop_path(x)
        x = x + residual
        return x

    def profile_module(
            self, input: Tensor, *args, **kwargs
        ) -> Tuple[Tensor, float, float]:
        b_sz, c, h, w = input.shape
        seq_len = h * w

        out, p_ffn, m_ffn = module_profile(module=self.mlp, x=input)
        # m_ffn = m_ffn * b_sz * seq_len

        out, p_mha, m_mha = module_profile(module=self.filter, x=out)


        macs = m_mha + m_ffn
        params = p_mha + p_ffn

        return input, params, macs




class AFFBlock(BaseModule):

    def __init__(
            self,
            opts,
            in_channels: int,
            transformer_dim: int,
            ffn_dim: int,
            n_transformer_blocks: Optional[int] = 2,
            head_dim: Optional[int] = 32,
            attn_dropout: Optional[float] = 0.0,
            dropout: Optional[int] = 0.0,
            ffn_dropout: Optional[int] = 0.0,
            patch_h: Optional[int] = 8,
            patch_w: Optional[int] = 8,
            attn_norm_layer: Optional[str] = "layer_norm_2d",
            conv_ksize: Optional[int] = 3,
            dilation: Optional[int] = 1,
            no_fusion: Optional[bool] = False,
            *args,
            **kwargs
    ) -> None:

        conv_1x1_out = ConvLayer(
            opts=opts,
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            use_norm=True,
            use_act=False,
        )
        conv_3x3_out = None
        if not no_fusion:
            conv_3x3_out = ConvLayer(
                opts=opts,
                in_channels=2 * in_channels,
                out_channels=in_channels,
                kernel_size=1,  # conv_ksize -> 1
                stride=1,
                use_norm=True,
                use_act=True,
            )
        super().__init__()

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim
        self.enable_coreml_compatible_fn = getattr(
            opts, "common.enable_coreml_compatible_module", False
        ) or getattr(opts, "benchmark.use_jit_model", False)
        print(self.enable_coreml_compatible_fn)

        global_rep = [
            # TODO: to check the double skip
            Block(
                opts=opts,
                dim=transformer_dim,
                hidden_size=transformer_dim,
                num_blocks=8,
                double_skip=False,
                mlp_ratio=ffn_dim / transformer_dim,
                attn_norm_layer=attn_norm_layer,
                enable_coreml_compatible_fn=self.enable_coreml_compatible_fn
            )
            for _ in range(n_transformer_blocks)
        ]
        global_rep.append(
            get_normalization_layer(
                opts=opts,
                norm_type=attn_norm_layer,
                num_features=transformer_dim,
            )
        )
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out

        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.dilation = dilation
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def __repr__(self) -> str:
        repr_str = "{}(".format(self.__class__.__name__)

        repr_str += "\n\t Global representations with patch size of {}x{}".format(
            self.patch_h, self.patch_w
        )
        if isinstance(self.global_rep, nn.Sequential):
            for m in self.global_rep:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.global_rep)

        if isinstance(self.conv_proj, nn.Sequential):
            for m in self.conv_proj:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.conv_proj)

        if self.fusion is not None:
            repr_str += "\n\t Feature fusion"
            if isinstance(self.fusion, nn.Sequential):
                for m in self.fusion:
                    repr_str += "\n\t\t {}".format(m)
            else:
                repr_str += "\n\t\t {}".format(self.fusion)

        repr_str += "\n)"
        return repr_str

    def forward_spatial(self, x: Tensor) -> Tensor:
        res = x

        # fm = self.local_rep(x)
        patches = x

        # b, c, h, w = fm.size()
        # patches = einops.rearrange(fm, 'b c h w -> b (h w) c')

        # learn global representations
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)

        # fm = einops.rearrange(patches, 'b (h w) c -> b c h w', h=h, w=w)

        fm = self.conv_proj(patches)

        if self.fusion is not None:
            fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm

    def forward_temporal(
            self, x: Tensor, x_prev: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        res = x
        fm = self.local_rep(x)

        # # convert feature map to patches
        # patches, info_dict = self.unfolding(fm)

        # learn global representations
        for global_layer in self.global_rep:
            if isinstance(global_layer, TransformerEncoder):
                patches = global_layer(x=patches, x_prev=x_prev)
            else:
                patches = global_layer(patches)

        # # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        # fm = self.folding(patches=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        if self.fusion is not None:
            fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm, patches

    def forward(
            self, x: Union[Tensor, Tuple[Tensor]], *args, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if isinstance(x, Tuple) and len(x) == 2:
            # for spatio-temporal MobileViT
            return self.forward_temporal(x=x[0], x_prev=x[1])
        elif isinstance(x, Tensor):
            # For image data
            return self.forward_spatial(x)
        else:
            raise NotImplementedError

    def profile_module(
            self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        params = macs = 0.0

        res = input

        b, c, h, w = input.size()

        out, p, m = module_profile(module=self.global_rep, x=input)
        params += p
        macs += m

        out, p, m = module_profile(module=self.conv_proj, x=out)
        params += p
        macs += m

        if self.fusion is not None:
            out, p, m = module_profile(
                module=self.fusion, x=torch.cat((out, res), dim=1)
            )
            params += p
            macs += m

        return res, params, macs

