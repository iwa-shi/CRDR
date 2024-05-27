"""
ELIC (CVPR2022)
- D.He et al. "ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding"

InterpCA (ACMMM2022)
- Z.Sun et al. "Interpolation Variable Rate Image Compression"

"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.registry import ENCODER_REGISTRY, DECODER_REGISTRY

from src.models.layer.interp_channel_attention import InterpChAtt
from .elic_autoencoder import ElicEncoder, ElicDecoder


@ENCODER_REGISTRY.register()
class ElicInterpCaEncoder(ElicEncoder):
    def __init__(self,
                 rate_level: int,
                 in_ch: int=3,
                 out_ch: int=192,
                 main_ch: int=192,
                 block_mid_ch: int=192,
                 num_blocks: int=3,
                 ca_kwargs: Dict={}):
        super().__init__(in_ch=in_ch, out_ch=out_ch, main_ch=main_ch, 
                         block_mid_ch=block_mid_ch, num_blocks=num_blocks)

        # Name of the layer and its output channel
        self.layer_out_ch_list = [
            ('conv1', main_ch),
            ('block1', main_ch),
            ('conv2', main_ch),
            ('block2', main_ch),
            ('attn2', main_ch),
            ('conv3', main_ch),
            ('block3', main_ch),
            ('conv4', out_ch),
            ('attn4', out_ch),
        ]
        self.interp_ca_list = nn.ModuleList()
        for _, _ch in self.layer_out_ch_list:
            self.interp_ca_list.append(InterpChAtt(_ch, rate_level, **ca_kwargs))

    def forward(self, x, rate_ind):
        for (layer_name, _), interp_ca in zip(self.layer_out_ch_list, self.interp_ca_list):
            layer = getattr(self, layer_name)
            x = layer(x) # layer -> interp_ca
            x = interp_ca(x, rate_ind)
        return x


@DECODER_REGISTRY.register()
class ElicInterpCaDecoder(ElicDecoder):
    def __init__(self,
                 rate_level: int,
                 in_ch: int=192,
                 out_ch: int=3,
                 main_ch: int=192,
                 block_mid_ch: int=192,
                 num_blocks: int=3,
                 use_tanh: bool=True,
                 pixel_shuffle: bool=False,
                 ca_kwargs: Dict={}):
        super().__init__(in_ch=in_ch, out_ch=out_ch, main_ch=main_ch, block_mid_ch=block_mid_ch,
                         num_blocks=num_blocks, use_tanh=use_tanh, pixel_shuffle=pixel_shuffle)

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

    def forward(self, x, rate_ind):
        for (layer_name, _), interp_ca in zip(self.layer_in_ch_list, self.interp_ca_list):
            layer = getattr(self, layer_name)
            x = interp_ca(x, rate_ind) # interp_ca -> layer
            x = layer(x)
        if self.use_tanh:
            x = torch.tanh(x)
        return x
