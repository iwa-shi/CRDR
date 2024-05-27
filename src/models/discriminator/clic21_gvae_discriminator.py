from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.discriminator.base_discriminator import BaseDiscriminator
from src.models.layer.hific_norm import ChannelNorm2D_wrap
from src.utils.registry import DISCRIMINATOR_REGISTRY

def conv_bn_lrelu(in_ch, out_ch, kernel_size=3, stride=1, norm_type: str='BN'):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)]
    if norm_type == 'BN':
        layers.append(nn.BatchNorm2d(out_ch))
    elif norm_type == 'IN':
        layers.append(nn.InstanceNorm2d(out_ch))
    elif norm_type == 'CN':
        layers.append(ChannelNorm2D_wrap(out_ch))
    elif norm_type == 'none':
        pass
    else:
        raise ValueError(f'Invalid norm_type: {norm_type}')
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers

def clic21_gvae_discriminator_blocks(in_ch, main_ch, out_ch, kw, norm_type='BN', num_downscale=4, head=True):
    layers = []
    layers.extend(conv_bn_lrelu(in_ch, main_ch, kernel_size=kw, stride=1, norm_type='none'))
    layers.extend(conv_bn_lrelu(main_ch, main_ch, kernel_size=kw, stride=2, norm_type=norm_type))

    _in_ch = main_ch
    for _ in range(num_downscale-1):
        _out_ch = min(_in_ch * 2, main_ch * 8)
        layers.extend(conv_bn_lrelu(_in_ch, _out_ch, kernel_size=kw, stride=1, norm_type=norm_type))
        layers.extend(conv_bn_lrelu(_out_ch, _out_ch, kernel_size=kw, stride=2, norm_type=norm_type))
        _in_ch = _out_ch
    if head:
        layers.append(nn.Conv2d(_in_ch, out_ch, kernel_size=3, stride=1, padding=1))
    return nn.Sequential(*layers)

# Defines the PatchGAN discriminator with the specified arguments.
@DISCRIMINATOR_REGISTRY.register()
class CLIC21GVAEDiscriminator(BaseDiscriminator):
    def __init__(self, in_ch=3, out_ch=1, main_ch=64, norm_type: str='BN', num_downscale: int=4):
        super(CLIC21GVAEDiscriminator, self).__init__()
        self.model = clic21_gvae_discriminator_blocks(in_ch, main_ch, out_ch, kw=3, norm_type=norm_type, num_downscale=num_downscale)

    def forward(self, input, **kwargs):
        return self.model(input)


@DISCRIMINATOR_REGISTRY.register()
class CLIC21GVAELatentConditionalDiscriminator(BaseDiscriminator):
    def __init__(self, in_ch=3, out_ch=1, y_ch=192, latent_nc=12, main_ch=64, norm_type: str='BN', latent_interp_mode: str='bilinear'):
        super(CLIC21GVAELatentConditionalDiscriminator, self).__init__()
        self.latent_conv = nn.Sequential(*conv_bn_lrelu(y_ch, latent_nc, kernel_size=1, stride=1, norm_type='none'))
        self.model = clic21_gvae_discriminator_blocks(in_ch+latent_nc, main_ch, out_ch, kw=3, norm_type=norm_type)

        assert latent_interp_mode in ['bilinear', 'bicubic', 'nearest']
        self.latent_interp_kwargs = dict(mode=latent_interp_mode)
        if latent_interp_mode != 'nearest':
            self.latent_interp_kwargs['align_corners'] = False

    def forward(self, input, y_hat, **kwargs):
        cond = self.latent_conv(y_hat.detach())
        cond = F.interpolate(cond, scale_factor=16, **self.latent_interp_kwargs)
        return self.model(torch.cat((input, cond), dim=1))
