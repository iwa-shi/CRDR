"""
Dalian He et al.
"ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding"
CVPR, 2022

https://arxiv.org/abs/2203.10886 (Details of network structure can be found in the Supplementary Material)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.subnet.autoencoder.base_autoencoder import BaseEncoder, BaseDecoder
from src.utils.registry import ENCODER_REGISTRY, DECODER_REGISTRY

from src.models.layer.cheng_nlam import ChengNLAM
from src.models.layer.elic_layers import ResidualBottleneckBlocks


def up_conv(in_ch: int, out_ch: int, kernel_size: int, pixel_shuffle: bool):
    assert kernel_size == 5, 'For now, we only support kernel_size=5, which is used in ELIC paper'
    if pixel_shuffle:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch*4, kernel_size, stride=1, padding=kernel_size//2),
            nn.PixelShuffle(2),
        )
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size, padding=2, stride=2, output_padding=1)


@ENCODER_REGISTRY.register()
class ElicEncoder(BaseEncoder):
    def __init__(self, 
                 in_ch: int=3,
                 out_ch: int=192,
                 main_ch: int=192,
                 block_mid_ch: int=192,
                 num_blocks: int=3,
                 res_in_res: bool=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, main_ch, kernel_size=5, stride=2, padding=2)
        self.block1 = ResidualBottleneckBlocks(main_ch, mid_ch=block_mid_ch, num_blocks=num_blocks, res_in_res=res_in_res)

        self.conv2 = nn.Conv2d(main_ch, main_ch, kernel_size=5, stride=2, padding=2)
        self.block2 = ResidualBottleneckBlocks(main_ch, mid_ch=block_mid_ch, num_blocks=num_blocks, res_in_res=res_in_res)
        self.attn2 = ChengNLAM(main_ch)

        self.conv3 = nn.Conv2d(main_ch, main_ch, kernel_size=5, stride=2, padding=2)
        self.block3 = ResidualBottleneckBlocks(main_ch, mid_ch=block_mid_ch, num_blocks=num_blocks, res_in_res=res_in_res)

        self.conv4 = nn.Conv2d(main_ch, out_ch, kernel_size=5, stride=2, padding=2)
        self.attn4 = ChengNLAM(out_ch)

        self.num_downscale = 4
        self.latent_ch = out_ch

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)

        x = self.conv2(x)
        x = self.block2(x)
        x = self.attn2(x)

        x = self.conv3(x)
        x = self.block3(x)

        x = self.conv4(x)
        x = self.attn4(x)

        return x


@DECODER_REGISTRY.register()
class ElicDecoder(BaseDecoder):
    def __init__(self, 
                 in_ch: int=192, 
                 out_ch: int=3, 
                 main_ch: int=192, 
                 block_mid_ch: int=192, 
                 num_blocks: int=3, 
                 use_tanh: bool=True, 
                 pixel_shuffle: bool=False,
                 res_in_res: bool=False):
        super().__init__()
        self.use_tanh = use_tanh

        self.attn1 = ChengNLAM(in_ch)
        self.conv1 = up_conv(in_ch, main_ch, kernel_size=5, pixel_shuffle=pixel_shuffle)
        self.block1 = ResidualBottleneckBlocks(main_ch, mid_ch=block_mid_ch, num_blocks=num_blocks, res_in_res=res_in_res)

        self.conv2 = up_conv(main_ch, main_ch, kernel_size=5, pixel_shuffle=pixel_shuffle)
        self.attn2 = ChengNLAM(main_ch)
        self.block2 = ResidualBottleneckBlocks(main_ch, mid_ch=block_mid_ch, num_blocks=num_blocks, res_in_res=res_in_res)

        self.conv3 = up_conv(main_ch, main_ch, kernel_size=5, pixel_shuffle=pixel_shuffle)
        self.block3 = ResidualBottleneckBlocks(main_ch, mid_ch=block_mid_ch, num_blocks=num_blocks, res_in_res=res_in_res)

        self.conv4 = up_conv(main_ch, out_ch, kernel_size=5, pixel_shuffle=pixel_shuffle)

    def forward(self, x):
        x = self.attn1(x)
        x = self.conv1(x)
        x = self.block1(x)

        x = self.conv2(x)
        x = self.attn2(x)
        x = self.block2(x)

        x = self.conv3(x)
        x = self.block3(x)

        x = self.conv4(x)

        if self.use_tanh:
            x = torch.tanh(x)

        return x
