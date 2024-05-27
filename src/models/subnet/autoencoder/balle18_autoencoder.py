import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.layers import GDN

from .base_autoencoder import BaseEncoder, BaseDecoder
from src.utils.registry import ENCODER_REGISTRY, DECODER_REGISTRY

@ENCODER_REGISTRY.register()
class Balle18Encoder(BaseEncoder):
    def __init__(self, in_ch=3, out_ch=192, main_ch=192):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, main_ch, kernel_size=5, stride=2, padding=2),
            GDN(main_ch),
            nn.Conv2d(main_ch, main_ch, kernel_size=5, stride=2, padding=2),
            GDN(main_ch),
            nn.Conv2d(main_ch, main_ch, kernel_size=5, stride=2, padding=2),
            GDN(main_ch),
            nn.Conv2d(main_ch, out_ch, kernel_size=5, stride=2, padding=2),
        )
        self.num_downscale = 4
        self.latent_ch = out_ch

    def forward(self, x):
        x = self.conv(x)
        return x


@DECODER_REGISTRY.register()
class Balle18Decoder(BaseDecoder):
    def __init__(self, in_ch=192, out_ch=3, main_ch=192, use_tanh: bool=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, main_ch, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(main_ch, inverse=True),
            nn.ConvTranspose2d(main_ch, main_ch, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(main_ch, inverse=True),
            nn.ConvTranspose2d(main_ch, main_ch, kernel_size=5, stride=2, padding=2, output_padding=1),
            GDN(main_ch, inverse=True),
            nn.ConvTranspose2d(main_ch, out_ch, kernel_size=5, stride=2, padding=2, output_padding=1),
        )
        self.use_tanh = use_tanh

    def forward(self, x):
        x = self.conv(x)
        # x = torch.clamp(x, min=0., max=1.)
        if self.use_tanh:
            x = torch.tanh(x)
        return x
