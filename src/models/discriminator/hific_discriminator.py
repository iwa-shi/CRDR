import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from src.models.discriminator.base_discriminator import BaseDiscriminator
from src.utils.registry import DISCRIMINATOR_REGISTRY

def conv2d(*args, use_sn: bool, **kwargs):
    if use_sn:
        return spectral_norm(nn.Conv2d(*args, **kwargs))
    return nn.Conv2d(*args, **kwargs)

def conv_lrelu(in_ch, out_ch, kernel_size=3, stride=1, use_sn: bool=True):
    padding = int(np.ceil((kernel_size-1.0)/2))
    return [
            conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, use_sn=use_sn), 
            nn.LeakyReLU(0.2, inplace=True)
    ]

# Defines the PatchGAN discriminator with the specified arguments.
@DISCRIMINATOR_REGISTRY.register()
class HiFiCDiscriminator(BaseDiscriminator):
    def __init__(self, in_ch=3, out_ch=1, main_ch=64, use_sn: bool=True, cond: bool=False):
        super().__init__()
        kw = 4
        layers = []
        layers.extend(conv_lrelu(in_ch, main_ch, kernel_size=kw, stride=2, use_sn=use_sn))
        layers.extend(conv_lrelu(main_ch, main_ch*2, kernel_size=kw, stride=2, use_sn=use_sn))
        layers.extend(conv_lrelu(main_ch*2, main_ch*4, kernel_size=kw, stride=2, use_sn=use_sn))
        layers.extend(conv_lrelu(main_ch*4, main_ch*8, kernel_size=kw, stride=1, use_sn=use_sn))
        layers.append(conv2d(main_ch*8, out_ch, kernel_size=1, stride=1, padding=0, use_sn=use_sn))
        self.model = nn.Sequential(*layers)

    def forward(self, input, **kwargs):
        return self.model(input)


@DISCRIMINATOR_REGISTRY.register()
class HiFiCConditionalDiscriminator(BaseDiscriminator):
    def __init__(self, in_ch=3, out_ch=1, main_ch=64, y_ch=192, latent_nc=12, use_sn: bool=True, cond: bool=False):
        super().__init__()
        kw = 4
        self.latent_conv = nn.Sequential(*conv_lrelu(y_ch, latent_nc, kernel_size=1, stride=1, use_sn=False))
        layers = []
        layers.extend(conv_lrelu(in_ch+latent_nc, main_ch, kernel_size=kw, stride=2, use_sn=use_sn))
        layers.extend(conv_lrelu(main_ch, main_ch*2, kernel_size=kw, stride=2, use_sn=use_sn))
        layers.extend(conv_lrelu(main_ch*2, main_ch*4, kernel_size=kw, stride=2, use_sn=use_sn))
        layers.extend(conv_lrelu(main_ch*4, main_ch*8, kernel_size=kw, stride=1, use_sn=use_sn))
        layers.append(conv2d(main_ch*8, out_ch, kernel_size=1, stride=1, padding=0, use_sn=use_sn))
        self.model = nn.Sequential(*layers)

    def forward(self, input, y_hat, **kwargs):
        cond = self.latent_conv(y_hat.detach())
        cond = F.interpolate(cond, scale_factor=16, mode='nearest')
        return self.model(torch.cat((input, cond), dim=1))
