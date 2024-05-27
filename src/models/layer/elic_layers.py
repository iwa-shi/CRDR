"""
Dalian He et al.
"ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding"
CVPR, 2022

https://arxiv.org/abs/2203.10886 (Details of network structure can be found in the Supplementary Material)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def up_conv(in_ch: int, out_ch: int, kernel_size: int, pixel_shuffle: bool):
    assert kernel_size == 5, 'For now, we only support kernel_size=5, which is used in ELIC paper'
    if pixel_shuffle:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch*4, kernel_size, stride=1, padding=kernel_size//2),
            nn.PixelShuffle(2),
        )
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size, padding=2, stride=2, output_padding=1)

class BaseBlock(nn.Module):
    def __init__(self, ch: int, mid_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, mid_ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, ch, kernel_size=1, stride=1, padding=0),
        )
    
    def forward(self, x):
        y = self.conv(x)
        return x + y

class ResidualBottleneckBlocks(nn.Module):
    def __init__(self, ch: int, mid_ch: int, num_blocks: int=3, res_in_res: bool=False):
        super().__init__()
        self.num_blocks = num_blocks
        for i in range(num_blocks):
            setattr(self, f'block{i}', BaseBlock(ch, mid_ch))
        self.use_residual = res_in_res

    def forward(self, x):
        y = x
        for i in range(self.num_blocks):
            block = getattr(self, f'block{i}')
            y = block(y)
        if self.use_residual:
            y = x + y
        return y