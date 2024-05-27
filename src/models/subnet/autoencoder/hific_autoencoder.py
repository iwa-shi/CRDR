"""
original code
    https://github.com/Justin-Tan/high-fidelity-generative-compression/blob/master/src/network/encoder.py
    https://github.com/Justin-Tan/high-fidelity-generative-compression/blob/master/src/network/generator.py

[1] Mentzer et. al., "High-Fidelity Generative Image Compression", arXiv:2006.09965 (2020).
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.subnet.autoencoder.base_autoencoder import BaseEncoder, BaseDecoder
from src.models.layer.hific_norm import ChannelNorm2D_wrap, InstanceNorm2D_wrap
from src.utils.registry import ENCODER_REGISTRY, DECODER_REGISTRY


@ENCODER_REGISTRY.register()
class HificEncoder(BaseEncoder):
    def __init__(self,
                 in_ch: int=3,
                 bottleneck_y: int=220,
                 filters: List=[60, 120, 240, 480, 960],
                 activation: str='relu',
                 use_norm=True,
                 channel_norm: bool=True):
        """HiFiC Encoder

        Args:
            in_ch (int, optional): Defaults to 3.
            bottleneck_y (int, optional): Defaults to 220.
            activation (str, optional): Defaults to 'relu'.
            channel_norm (bool, optional): Defaults to True.
        """
        
        super(HificEncoder, self).__init__()
        
        kernel_dim = 3

        # Layer / normalization options
        cnn_kwargs = dict(stride=2, padding=0, padding_mode='reflect')
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)
        activation_d = dict(relu='ReLU', elu='ELU', leaky_relu='LeakyReLU')
        self.activation = getattr(nn, activation_d[activation])  # (leaky_relu, relu, elu)
        self.n_downsampling_layers = 4

        if not use_norm:
            self.interlayer_norm = nn.Identity
        elif channel_norm is True:
            self.interlayer_norm = ChannelNorm2D_wrap
        else:
            self.interlayer_norm = InstanceNorm2D_wrap

        self.pre_pad = nn.ReflectionPad2d(3)
        self.asymmetric_pad = nn.ReflectionPad2d((0,1,1,0))  # Slower than tensorflow?
        self.post_pad = nn.ReflectionPad2d(1)

        # (256,256) -> (256,256), with implicit padding
        self.conv_block1 = nn.Sequential(
            self.pre_pad,
            nn.Conv2d(in_ch, filters[0], kernel_size=(7,7), stride=1),
            self.interlayer_norm(filters[0], **norm_kwargs),
            self.activation(),
        )

        # (256,256) -> (128,128)
        self.conv_block2 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(filters[0], filters[1], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[1], **norm_kwargs),
            self.activation(),
        )

        # (128,128) -> (64,64)
        self.conv_block3 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(filters[1], filters[2], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[2], **norm_kwargs),
            self.activation(),
        )

        # (64,64) -> (32,32)
        self.conv_block4 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(filters[2], filters[3], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[3], **norm_kwargs),
            self.activation(),
        )

        # (32,32) -> (16,16)
        self.conv_block5 = nn.Sequential(
            self.asymmetric_pad,
            nn.Conv2d(filters[3], filters[4], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[4], **norm_kwargs),
            self.activation(),
        )
        
        # Project channels onto space w/ dimension C
        # Feature maps have dimension C x W/16 x H/16
        # (16,16) -> (16,16)
        self.conv_block_out = nn.Sequential(
            self.post_pad,
            nn.Conv2d(filters[4], bottleneck_y, kernel_dim, stride=1),
        )

        self.num_downscale = 4
        self.latent_ch = bottleneck_y
        

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        out = self.conv_block_out(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_ch,
                 kernel_size=3,
                 stride=1, 
                 use_norm=True,
                 channel_norm=True,
                 activation='relu'):
        """
        input_dims: Dimension of input tensor (B,C,H,W)
        """
        super(ResidualBlock, self).__init__()

        self.activation = getattr(F, activation)
        in_channels = in_ch
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)
        
        if not use_norm:
            self.interlayer_norm = nn.Identity
        elif channel_norm is True:
            self.interlayer_norm = ChannelNorm2D_wrap
        else:
            self.interlayer_norm = InstanceNorm2D_wrap

        pad_size = int((kernel_size-1)/2)
        self.pad = nn.ReflectionPad2d(pad_size)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride)
        self.norm1 = self.interlayer_norm(in_channels, **norm_kwargs)
        self.norm2 = self.interlayer_norm(in_channels, **norm_kwargs)

    def forward(self, x):
        identity_map = x
        res = self.pad(x)
        res = self.conv1(res)
        res = self.norm1(res) 
        res = self.activation(res)

        res = self.pad(res)
        res = self.conv2(res)
        res = self.norm2(res)

        return torch.add(res, identity_map)


class ConvPixelShuffle(nn.Module):
    def __init__(self, cin: int, cout: int, kernel_size: int, **kwargs) -> None:
        super().__init__()
        # self.conv1 = nn.Conv2d(cin, cin, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        # self.conv2 = nn.Conv2d(cin, cout*4, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(cin, cout*4, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv1(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        x = self.ps(x)
        return x


@DECODER_REGISTRY.register()
class HificDecoder(BaseDecoder):
    def __init__(self,
                 bottleneck_y=220,
                 activation='relu',
                 n_residual_blocks=9,
                 filters: List= [960, 480, 240, 120, 60],
                 use_norm=True,
                 channel_norm=True,
                 use_first_norm=True,
                 sample_noise=False,
                 use_tanh=True,
                 use_pixelshuffle=False,
                 noise_dim=32):
        super(HificDecoder, self).__init__()
        
        kernel_dim = 3
        self.n_residual_blocks = n_residual_blocks
        self.sample_noise = sample_noise
        self.noise_dim = noise_dim

        # Layer / normalization options
        cnn_kwargs = dict(stride=2, padding=1, output_padding=1)
        norm_kwargs = dict(momentum=0.1, affine=True, track_running_stats=False)
        activation_d = dict(relu='ReLU', elu='ELU', leaky_relu='LeakyReLU')
        self.activation = getattr(nn, activation_d[activation])  # (leaky_relu, relu, elu)
        self.n_upsampling_layers = 4
        
        if not use_norm:
            self.interlayer_norm = nn.Identity
        elif channel_norm is True:
            self.interlayer_norm = ChannelNorm2D_wrap
        else:
            self.interlayer_norm = InstanceNorm2D_wrap

        self.pre_pad = nn.ReflectionPad2d(1)
        self.asymmetric_pad = nn.ReflectionPad2d((0,1,1,0))  # Slower than tensorflow?
        self.post_pad = nn.ReflectionPad2d(3)
        
        first_norm = self.interlayer_norm(bottleneck_y, **norm_kwargs) if use_first_norm else nn.Identity()

        up_layer = ConvPixelShuffle if use_pixelshuffle else nn.ConvTranspose2d

        # (16,16) -> (16,16), with implicit padding
        self.conv_block_init = nn.Sequential(
            first_norm,
            self.pre_pad,
            nn.Conv2d(bottleneck_y, filters[0], kernel_size=(3,3), stride=1),
            self.interlayer_norm(filters[0], **norm_kwargs),
        )

        if sample_noise is True:
            # Concat noise with latent representation
            filters[0] += self.noise_dim

        for m in range(n_residual_blocks):
            resblock_m = ResidualBlock(in_ch=filters[0], use_norm=use_norm,
                channel_norm=channel_norm, activation=activation)
            self.add_module(f'resblock_{str(m)}', resblock_m)
        
        # (16,16) -> (32,32)
        self.upconv_block1 = nn.Sequential(
            up_layer(filters[0], filters[1], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[1], **norm_kwargs),
            self.activation(),
        )

        self.upconv_block2 = nn.Sequential(
            up_layer(filters[1], filters[2], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[2], **norm_kwargs),
            self.activation(),
        )

        self.upconv_block3 = nn.Sequential(
            up_layer(filters[2], filters[3], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[3], **norm_kwargs),
            self.activation(),
        )

        self.upconv_block4 = nn.Sequential(
            up_layer(filters[3], filters[4], kernel_dim, **cnn_kwargs),
            self.interlayer_norm(filters[4], **norm_kwargs),
            self.activation(),
        )

        self.conv_block_out = nn.Sequential(
            self.post_pad,
            nn.Conv2d(filters[-1], 3, kernel_size=(7,7), stride=1),
        )
        self.use_tanh = use_tanh


    def forward(self, x):
        
        head = self.conv_block_init(x)

        if self.sample_noise is True:
            B, _, H, W = head.size()
            z = torch.randn((B, self.noise_dim, H, W)).to(head)
            head = torch.cat((head, z), dim=1)

        for m in range(self.n_residual_blocks):
            resblock_m = getattr(self, f'resblock_{str(m)}')
            if m == 0:
                x = resblock_m(head)
            else:
                x = resblock_m(x)
        
        x += head
        x = self.upconv_block1(x)
        x = self.upconv_block2(x)
        x = self.upconv_block3(x)
        x = self.upconv_block4(x)
        out = self.conv_block_out(x)
        if self.use_tanh:
            out = torch.tanh(out)

        return out