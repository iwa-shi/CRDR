"""
ELIC + InterpCA + BetaConditioning (Multi-realism)

ELIC (CVPR2022)
- D.He et al. "ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding"

InterpCA (ACMMM2022)
- Z.Sun et al. "Interpolation Variable Rate Image Compression"

BetaConditioning (CVPR2023)
- Agustsson et al. "Multi-Realism Image Compression with a Conditional Generator"

"""

from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from src.utils.registry import ENCODER_REGISTRY, DECODER_REGISTRY

from src.models.layer.interp_channel_attention import InterpChAtt
from src.models.subnet.autoencoder.base_autoencoder import  BaseDecoder
from src.models.layer.cheng_nlam import ChengNLAM
from src.models.layer.elic_layers import up_conv
from src.models.layer.fourier_cond import FourierEmbedding

def weights_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        module.weight.data.normal_(0.0, 0.02)
        module.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        module.weight.data.normal_(0.0, 0.02)
        module.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)

class BetaCondBaseBlock(nn.Module):
    def __init__(self, ch: int, mid_ch: int, cond_ch) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, mid_ch, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, ch, kernel_size=1, stride=1, padding=0),
        )
        self.proj_1 = nn.Conv2d(cond_ch, mid_ch, kernel_size=1, stride=1, padding=0)
        self.proj_2 = nn.Conv2d(cond_ch, mid_ch, kernel_size=1, stride=1, padding=0)
        self.proj_3 = nn.Conv2d(cond_ch, ch, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x: Tensor, cond_feat: Tensor) -> Tensor:
        sc = x
        x = self.conv[0](x)
        x = self.conv[1](x)
        x = x + self.proj_1(cond_feat)
        x = self.conv[2](x)
        x = self.conv[3](x)
        x = x + self.proj_2(cond_feat)
        x = self.conv[4](x)
        x = x + self.proj_3(cond_feat)
        return x + sc


class BetaCondResidualBottleneckBlocks(nn.Module):
    def __init__(self, ch: int, mid_ch: int, cond_ch: int, num_blocks: int=3, res_in_res: bool=False):
        super().__init__()
        self.num_blocks = num_blocks
        for i in range(num_blocks):
            setattr(self, f'block{i}', BetaCondBaseBlock(ch, mid_ch, cond_ch))
        self.use_residual = res_in_res

    def forward(self, x, cond_feat):
        y = x
        for i in range(self.num_blocks):
            block = getattr(self, f'block{i}')
            y = block(y, cond_feat)
        if self.use_residual:
            y = x + y
        return y


@DECODER_REGISTRY.register()
class ElicInterpCaBetaCondDecoder(BaseDecoder):
    def __init__(self,
                 rate_level: int,
                 L: int =10,
                 max_beta: float=5.12,
                 cond_ch: int=512,
                 use_pi: bool=True,
                 include_x: bool=False,
                 weight_init: bool=False,
                 in_ch: int=192,
                 out_ch: int=3,
                 main_ch: int=192,
                 block_mid_ch: int=192,
                 num_blocks: int=3,
                 use_tanh: bool=True,
                 pixel_shuffle: bool=False,
                 res_in_res: bool=False,
                 ca_kwargs: Dict={}):
        super().__init__()

        self.use_tanh = use_tanh

        self.attn1 = ChengNLAM(in_ch)
        self.conv1 = up_conv(in_ch, main_ch, kernel_size=5, pixel_shuffle=pixel_shuffle)
        self.block1 = BetaCondResidualBottleneckBlocks(main_ch, mid_ch=block_mid_ch, cond_ch=cond_ch, num_blocks=num_blocks, res_in_res=res_in_res)

        self.conv2 = up_conv(main_ch, main_ch, kernel_size=5, pixel_shuffle=pixel_shuffle)
        self.attn2 = ChengNLAM(main_ch)
        self.block2 = BetaCondResidualBottleneckBlocks(main_ch, mid_ch=block_mid_ch, cond_ch=cond_ch, num_blocks=num_blocks, res_in_res=res_in_res)

        self.conv3 = up_conv(main_ch, main_ch, kernel_size=5, pixel_shuffle=pixel_shuffle)
        self.block3 = BetaCondResidualBottleneckBlocks(main_ch, mid_ch=block_mid_ch, cond_ch=cond_ch, num_blocks=num_blocks, res_in_res=res_in_res)

        self.conv4 = up_conv(main_ch, out_ch, kernel_size=5, pixel_shuffle=pixel_shuffle)

        # name of the layer and its input channel
        self.layer_in_ch_list = [
            ('attn1', in_ch),
            ('conv1', in_ch),
            ('block1', main_ch),
            ('conv2', main_ch),
            ('attn2', main_ch),
            ('block2', main_ch),
            ('conv3', main_ch),
            ('block3', main_ch),
            ('conv4', main_ch),
        ]
        self.interp_ca_list = nn.ModuleList()
        for _, ch in self.layer_in_ch_list:
            self.interp_ca_list.append(InterpChAtt(ch, rate_level, **ca_kwargs))

        ## Beta Conditioning
        self.embed = FourierEmbedding(L=L, max_beta=max_beta, use_pi=use_pi, include_x=include_x)
        enc_ch = 2 * L + 1 if include_x else 2 * L
        self.mlp = nn.Sequential(
            nn.Linear(enc_ch, cond_ch),
            nn.ReLU(inplace=True),
            nn.Linear(cond_ch, cond_ch),
        )
        if weight_init:
            self.apply(weights_init)

    def forward(self, x: Tensor, rate_ind: Union[Tensor, float], beta: float) -> Tensor:
        cond = self.embed.embed(beta).detach().to(x.device) # [1, 2L]
        cond = self.mlp(cond).unsqueeze(-1).unsqueeze(-1) # [1, cond_ch, 1, 1]
        for (layer_name, _), interp_ca in zip(self.layer_in_ch_list, self.interp_ca_list):
            layer = getattr(self, layer_name)
            x = interp_ca(x, rate_ind) # interp_ca -> layer
            if 'block' in layer_name:
                x = layer(x, cond)
            else:
                x = layer(x)
        if self.use_tanh:
            x = torch.tanh(x)
        return x

