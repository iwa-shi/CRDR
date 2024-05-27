from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.layers import GDN

def get_actv(actv: str, ch: Optional[int]=None):
    actv_dict = dict(
        relu  = nn.ReLU(inplace=True),
        lrelu = nn.LeakyReLU(0.2, inplace=True),
        gdn   = GDN(ch), 
        igdn  = GDN(ch, inverse=True),
    )
    return actv_dict[actv]


class ResBlock(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 bn: bool=False,
                 actv: str='relu',
                 actv2: Optional[str]=None,
                 downscale: bool=False,
                 kernel_size: int=3,
                 padding_mode: str='zeros'):
        super().__init__()
        stride = 2 if downscale else 1
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=pad, stride=stride, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=pad, stride=1, padding_mode=padding_mode)

        self.actv1 = None if actv is None else get_actv(actv, out_channel)
        self.actv2 = None if actv2 is None else get_actv(actv2, out_channel)
        
        if bn:
            self.bn1 = nn.BatchNorm2d(out_channel)
            self.bn2 = nn.BatchNorm2d(out_channel)
        self.bn = bn

        if downscale or (in_channel != out_channel):
            self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = x
        if self.shortcut:
            shortcut = self.shortcut(shortcut)
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)
        if self.actv1:
            x = self.actv1(x)
        x = self.conv2(x)
        if self.bn:
            x = self.bn2(x)
        if self.actv2:
            x = self.actv2(x)
        x = x + shortcut
        return x


def upconv(in_ch: int, out_ch: int, kernel_size: int, up_mode: str, padding_mode: str='zeros'):
    conv_kwargs = dict(padding=(kernel_size - 1) // 2, padding_mode=padding_mode)
    if up_mode == 'pixelshuffle':
        return [
            nn.Conv2d(in_ch, out_ch*4, kernel_size=kernel_size, **conv_kwargs),
            nn.PixelShuffle(2),
        ]
    if up_mode == 'interpolate':
        return [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, **conv_kwargs),
            nn.Upsample(scale_factor=2, mode='nearest'),
        ]
    raise ValueError(f'Invalid up_mode: {up_mode}')


class UpResBlock(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_size: int=3,
                 actv: str='relu',
                 actv2: Optional[str]=None,
                 up_type: str='pixelshuffle',
                 padding_mode: str='zeros'):
        super().__init__()
        conv_kwargs = dict(padding=(kernel_size - 1) // 2, padding_mode=padding_mode)

        self.c1 = nn.Sequential(
            *upconv(in_channel, out_channel, kernel_size, up_mode=up_type, padding_mode=padding_mode),
            get_actv(actv, out_channel),
            nn.Conv2d(out_channel, out_channel, kernel_size, **conv_kwargs),
            get_actv(actv2, out_channel) if actv2 else nn.Identity(),
        )
        self.shortcut = nn.Sequential(
            *upconv(in_channel, out_channel, 1, up_mode=up_type, padding_mode=padding_mode),
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.c1(x)
        return x + shortcut
