import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_hyperprior import BaseHyperEncoder, BaseHyperDecoder
from src.utils.registry import HYPERENCODER_REGISTRY, HYPERDECODER_REGISTRY

@HYPERENCODER_REGISTRY.register()
class Balle18HyperEncoder(BaseHyperEncoder):
    def __init__(self, in_ch=192, out_ch=192, main_ch=192):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, main_ch, kernel_size=3, padding=1, stride=1)
        self.c2 = nn.Conv2d(main_ch, main_ch, kernel_size=5, padding=2, stride=2)
        self.c3 = nn.Conv2d(main_ch, out_ch, kernel_size=5, padding=2, stride=2)
        self.actv = nn.ReLU(inplace=True)
        
        self.num_downscale = 2
        self.latent_ch = out_ch

    def forward(self, x):
        x = self.c1(x)
        x = self.actv(x)
        x = self.c2(x)
        x = self.actv(x)
        x = self.c3(x)
        return x


@HYPERDECODER_REGISTRY.register()
class Balle18HyperDecoder(BaseHyperDecoder):
    def __init__(self, in_ch=192, out_ch=384, main_ch=192):
        super().__init__()
        self.c1 = nn.ConvTranspose2d(in_ch, main_ch, kernel_size=5, padding=2, stride=2, output_padding=1)
        self.c2 = nn.ConvTranspose2d(main_ch, main_ch, kernel_size=5, padding=2, stride=2, output_padding=1)
        self.c3 = nn.Conv2d(main_ch, out_ch, kernel_size=3, padding=1, stride=1)
        self.actv = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.c1(x)
        x = self.actv(x)
        x = self.c2(x)
        x = self.actv(x)
        x = self.c3(x)
        return x
