"""
Z.Sun et al. "Interpolation Variable Rate Image Compression", ACMMM2022
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.registry import ENCODER_REGISTRY, DECODER_REGISTRY
from src.models.layer.interp_channel_attention import InterpChAtt
from src.models.subnet.autoencoder.cheng20_autoencoder import Cheng20Encoder, Cheng20Decoder


@ENCODER_REGISTRY.register()
class Cheng20InterpCaEncoder(Cheng20Encoder):
    def __init__(self,
                 rate_level: int,
                 in_ch: int=3,
                 out_ch: int=192,
                 main_ch: int=192,
                 padding_mode: str='zeros',
                 ca_kwargs: Dict={},
                 **kwargs):
        super().__init__(in_ch=in_ch, out_ch=out_ch, main_ch=main_ch, padding_mode=padding_mode)
        self.layer_list = [
            'block1','block2', 'block3',
            'nlam1', 'block4', 'block5',
            'block6', 'conv7', 'nlam2'
        ]
        self.interp_ca_list = nn.ModuleList()
        for _ in range(len(self.layer_list)):
            self.interp_ca_list.append(InterpChAtt(main_ch, rate_level, **ca_kwargs))

    def forward(self, x, rate_ind):
        for layer_name, interp_ca in zip(self.layer_list, self.interp_ca_list):
            layer = getattr(self, layer_name)
            x = layer(x) # layer -> interp_ca
            x = interp_ca(x, rate_ind)
        return x


@DECODER_REGISTRY.register()
class Cheng20InterpCaDecoder(Cheng20Decoder):
    def __init__(self,
                rate_level: int,
                in_ch: int=192,
                out_ch: int=3,
                main_ch: int=192,
                use_tanh: bool=True,
                padding_mode: str='zeros',
                ca_kwargs: Dict={},
                **kwargs):
        super().__init__(in_ch=in_ch, out_ch=out_ch, main_ch=main_ch, use_tanh=use_tanh, padding_mode=padding_mode)
        self.layer_list = [
            'nlam0', 'block0', 'up0', 
            'block1', 'up1', 
            'nlam2', 'block2', 'up2', 
            'block3', 'up3', 
        ]
        self.interp_ca_list = nn.ModuleList()
        for i in range(len(self.layer_list)):
            ch = in_ch if i < 2 else main_ch
            self.interp_ca_list.append(InterpChAtt(ch, rate_level, **ca_kwargs))

    def forward(self, x, rate_ind):
        for layer_name, interp_ca in zip(self.layer_list, self.interp_ca_list):
            layer = getattr(self, layer_name)
            x = interp_ca(x, rate_ind) # interp_ca -> layer
            x = layer(x)
        if self.use_tanh:
            x = torch.tanh(x)
        return x

