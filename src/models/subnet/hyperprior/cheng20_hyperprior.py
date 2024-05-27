import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.subnet.hyperprior.base_hyperprior import BaseHyperEncoder, BaseHyperDecoder
from src.utils.registry import HYPERENCODER_REGISTRY, HYPERDECODER_REGISTRY

def up_conv(in_ch, out_ch, kernel_size=5):
    pad = 2 if kernel_size == 5 else 1
    out_pad = 1 if kernel_size == 5 else 0
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, stride=2, output_padding=out_pad),
        nn.LeakyReLU(0.2, inplace=True),
    )

def conv_lrelu(in_ch, out_ch, kernel_size=3, stride=1):
    return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2, stride=stride),
            nn.LeakyReLU(0.2, inplace=True),
        )

@HYPERENCODER_REGISTRY.register()
class Cheng20HyperEncoder(BaseHyperEncoder):
    def __init__(self, in_ch=192, out_ch=192, main_ch=192, **kwargs):
        super().__init__()
        self.c1 = conv_lrelu(in_ch, main_ch, kernel_size=3)
        self.c2 = conv_lrelu(main_ch, main_ch, kernel_size=3)
        self.c3 = conv_lrelu(main_ch, main_ch, kernel_size=3, stride=2)
        self.c4 = conv_lrelu(main_ch, main_ch, kernel_size=3)
        self.c5 = nn.Conv2d(main_ch, out_ch, kernel_size=3, padding=1, stride=2)
        self.num_downscale = 2
        self.latent_ch = out_ch

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        return x


@HYPERDECODER_REGISTRY.register()
class Cheng20HyperDecoder(BaseHyperDecoder):
    def __init__(self, in_ch=192, out_ch=384, main_ch=192, **kwargs):
        super().__init__()
        self.c1 = conv_lrelu(in_ch, main_ch, kernel_size=3)
        self.c2 = up_conv(main_ch, main_ch, kernel_size=4)
        self.c3 = conv_lrelu(main_ch, main_ch, kernel_size=3)
        self.c4 = up_conv(main_ch, main_ch, kernel_size=4)
        self.c5 = nn.Conv2d(main_ch, out_ch, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        return x