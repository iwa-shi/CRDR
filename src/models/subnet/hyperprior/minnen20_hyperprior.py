import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_hyperprior import BaseHyperEncoder, BaseHyperDecoder
from src.utils.registry import HYPERENCODER_REGISTRY, HYPERDECODER_REGISTRY

@HYPERENCODER_REGISTRY.register()
class Minnen20HyperEncoder(BaseHyperEncoder):
    def __init__(self, bottleneck_y: int=320, bottleneck_z: int=192):
        super().__init__()
        cnn_kwargs = dict(kernel_size=5, stride=2, padding=2)
        self.activation = nn.ReLU()
        self.n_downsampling_layers = 2

        self.conv1 = nn.Conv2d(bottleneck_y, 320, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(320, 256, **cnn_kwargs)
        self.conv3 = nn.Conv2d(256, bottleneck_z, **cnn_kwargs)

        self.num_downscale = 2
        self.latent_ch = bottleneck_z

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)
        return x


@HYPERDECODER_REGISTRY.register()
class Minnen20HyperDecoder(BaseHyperDecoder):
    def __init__(self, bottleneck_z: int=192, hyper_out_ch: int=640):
        super().__init__()
        assert hyper_out_ch % 2 == 0
        self.hd_mu = HyperDecoderBlock(in_ch=bottleneck_z, out_ch=hyper_out_ch//2)
        self.hd_std = HyperDecoderBlock(in_ch=bottleneck_z, out_ch=hyper_out_ch//2)

    def forward(self, x):
        mu = self.hd_mu(x)
        std = self.hd_std(x)
        return torch.cat([mu, std], dim=1)


class HyperDecoderBlock(nn.Module):
    def __init__(self, in_ch=192, out_ch=320):
        super(HyperDecoderBlock, self).__init__()

        cnn_kwargs = dict(kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation = nn.ReLU()
        self.conv1 = nn.ConvTranspose2d(in_ch, 192, **cnn_kwargs)
        self.conv2 = nn.ConvTranspose2d(192, 256, **cnn_kwargs)
        self.conv3 = nn.ConvTranspose2d(256, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)
        return x