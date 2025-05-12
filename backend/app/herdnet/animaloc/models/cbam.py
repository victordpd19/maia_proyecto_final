__copyright__ = \
    """
    Copyright (C) 2025 Victor Perez - Universidad de los Andes
    
    Based on original work:
    Copyright (c) 2024, Alexander Delplanque, University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.
    """
__authors__ = ["Victor Perez", "Jordi Sanchez, Maryi Carvajal, Simon Aristizabal"]
__license__ = "MIT"

""" CBAM (Convolutional Block Attention Module) adapted implementation from original
paper: https://arxiv.org/abs/1807.06521"""

import math
from os.path import join
from posixpath import basename

import torch
from torch import nn
import matplotlib.pyplot as plt

import numpy as np


# CBAM components (simplified)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (B, C, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(x_cat))
        return out

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, spatial_kernel=7, use_residual=False):
        super().__init__()
        self.channel_attention = ChannelAttention(gate_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel)
        self.use_residual = use_residual

    def forward(self, x):
        ca = self.channel_attention(x)
        sa = self.spatial_attention(x * ca)
        out = x * ca * sa
        return x + out if self.use_residual else out

