import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.layers import GDN

def get_actv(actv, ch=None):
    if actv == 'relu':
        return nn.ReLU(inplace=True)
    if actv == 'lrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    if actv == 'gdn':
        return GDN(ch)
    if actv == 'igdn':
        return GDN(ch, inverse=True)
    raise NotImplementedError(f'Invalid actv: "{actv}"')


class ResBlock(nn.Module):
    def __init__(self, in_channel=192, out_channel=192, bn=False, actv='relu', actv2=None, downscale=False, kernel_size=3):
        super().__init__()
        stride = 2 if downscale else 1
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=1, stride=1)

        self.actv1 = None if actv is None else get_actv(actv, out_channel)
        self.actv2 = None if actv2 is None else get_actv(actv2, out_channel)
        
        if bn:
            self.bn1 = nn.BatchNorm2d(out_channel)
            self.bn2 = nn.BatchNorm2d(out_channel)
        self.bn = bn

        if downscale:
            self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2)
        elif in_channel != out_channel:
            self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1)
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


class UpResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, actv='relu', actv2=None, up_type='pixelshuffle'):
        super().__init__()
        pad = (kernel_size - 1) // 2
        
        if up_type == 'pixelshuffle':
            main_layers = [
                nn.Conv2d(in_channel, out_channel*4, kernel_size=kernel_size, padding=pad),
                nn.PixelShuffle(2),
                get_actv(actv, out_channel*4),
                nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=pad),
            ]
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel*4, kernel_size=1),
                nn.PixelShuffle(2),
            )
        elif up_type == 'interpolate':
            main_layers = [
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=pad),
                nn.Upsample(scale_factor=2, mode='nearest'),
                get_actv(actv, out_channel),
                nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=pad),
            ]
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='nearest'),
            )
        else:
            raise NotImplementedError(f'Invalid up_type: "{up_type}"')

        if actv2 is not None:
            main_layers.append(get_actv(actv2, out_channel))

        self.c1 = nn.Sequential(*main_layers)
    
    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.c1(x)
        return x + shortcut
