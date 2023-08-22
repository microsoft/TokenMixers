import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from affnet.layers import ConvLayer, get_activation_fn, get_normalization_layer


# class BasicConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU() if relu else None
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class View_as_complex(nn.Module):
    def forward(self, x):
        return torch.view_as_complex(x)

class View_as_real(nn.Module):
    def forward(self, x):
        return torch.view_as_real(x)

class ChannelGateComplex(nn.Module):

    def __init__(self, opts, gate_channels, reduction_ratio=16):
        super(ChannelGateComplex, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Conv2d(gate_channels, gate_channels // reduction_ratio, kernel_size=1, dtype=torch.complex64),
            View_as_real(),
            self.build_act_layer(opts=opts),
            View_as_complex(),
            nn.Conv2d(gate_channels // reduction_ratio, gate_channels, kernel_size=1, dtype=torch.complex64)
        )

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

    def forward(self, x):
        # input complex
        input = x
        x = torch.view_as_real(x)
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.amax(x, dim=[2, 3], keepdim=True)
        avg_pool = torch.view_as_complex(avg_pool)
        max_pool = torch.view_as_complex(max_pool)
        avg_pool = self.mlp(avg_pool)
        max_pool = self.mlp(max_pool)
        channel_att_sum = torch.view_as_real(avg_pool + max_pool)
        output = torch.view_as_complex(F.sigmoid(channel_att_sum)) * input

        return output

class ChannelGate(nn.Module):

    def __init__(self, opts, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            self.build_act_layer(opts=opts),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

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

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class ChannelPoolComplex(nn.Module):
    def forward(self, x):
        x = torch.view_as_real(x)
        x = torch.cat( (torch.max(x, 1, keepdim=True)[0], torch.mean(x, 1, keepdim=True)), dim=1)
        return torch.view_as_complex(x)

class SpatialGateComplex(nn.Module):
    def __init__(self, opts):
        super(SpatialGateComplex, self).__init__()
        kernel_size = 7
        self.compress = ChannelPoolComplex()
        # self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1) // 2,
                                 dtype=torch.complex64)
        self.spatial_bn = nn.BatchNorm3d(1, eps=1e-5, momentum=0.01, affine=True)
    def forward(self, x):
        input = x
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        x_out = torch.view_as_real(x_out)
        x_out = self.spatial_bn(x_out)
        scale = F.sigmoid(x_out) # broadcasting
        scale = torch.view_as_complex(scale)
        return input * scale
class SpatialGate(nn.Module):
    def __init__(self, opts):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        # self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.spatial = ConvLayer(opts=opts, in_channels=2, out_channels=1, kernel_size=kernel_size,
                                 stride=1, padding=(kernel_size-1) // 2, use_norm=True, use_act=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):

    def __init__(self, opts, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False, no_channel=False):
        super(CBAM, self).__init__()
        self.no_channel = no_channel
        if not no_channel:
            self.ChannelGate = ChannelGate(opts, gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(opts=opts)

    def forward(self, x):
        if not self.no_channel:
            x_out = self.ChannelGate(x)
        else:
            x_out = x
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class CBAMComplex(nn.Module):

    def __init__(self, opts, gate_channels, reduction_ratio=16, no_spatial=False):
        super(CBAMComplex, self).__init__()
        self.ChannelGate = ChannelGateComplex(opts, gate_channels, reduction_ratio)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGateComplex(opts=opts)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out