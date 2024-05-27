import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.subnet.autoencoder.base_autoencoder import BaseEncoder, BaseDecoder
from src.utils.registry import ENCODER_REGISTRY, DECODER_REGISTRY

from src.models.layer.cheng_nlam import ChengNLAM
from src.models.layer.cheng_resblock import ResBlock, UpResBlock


@ENCODER_REGISTRY.register()
class Cheng20Encoder(BaseEncoder):
    def __init__(self,
                 in_ch: int=3,
                 out_ch: int=192,
                 main_ch: int=192,
                 padding_mode: str='zeros',
                 **kwargs):
        super().__init__()
        down_block_kwargs = dict(actv='lrelu', actv2='gdn', downscale=True, padding_mode=padding_mode)
        normal_block_kwargs = dict(actv='lrelu', actv2='lrelu', downscale=False, padding_mode=padding_mode)

        ## (H, W) -> (H/2, W/2)
        self.block1 = ResBlock(in_ch, main_ch, **down_block_kwargs)

        ## (H/2, W/2) -> (H/4, W/4)
        self.block2 = ResBlock(main_ch, main_ch, **normal_block_kwargs)
        self.block3 = ResBlock(main_ch, main_ch, **down_block_kwargs)

        ## (H/4, W/4) -> (H/8, W/8)
        self.nlam1  = ChengNLAM(main_ch, padding_mode=padding_mode)
        self.block4 = ResBlock(main_ch, main_ch, **normal_block_kwargs)
        self.block5 = ResBlock(main_ch, main_ch, **down_block_kwargs)

        ## (H/8, W/8) -> (H/16, W/16)
        self.block6 = ResBlock(main_ch, main_ch, **normal_block_kwargs)
        self.conv7  = nn.Conv2d(main_ch, out_ch, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode)

        ## (H/16, W/16)
        self.nlam2  = ChengNLAM(out_ch, padding_mode=padding_mode)
        self.num_downscale = 4
        self.latent_ch = out_ch

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.nlam1(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.conv7(x)
        x = self.nlam2(x)
        return x


@DECODER_REGISTRY.register()
class Cheng20Decoder(BaseDecoder):
    def __init__(self,
                 in_ch: int=192,
                 out_ch: int=3,
                 main_ch: int=192,
                 use_tanh: bool=True,
                 padding_mode: str='zeros',
                 **kwargs):
        super().__init__()
        up_block_kwargs = dict(actv='lrelu', actv2='igdn', padding_mode=padding_mode)
        normal_block_kwargs = dict(actv='lrelu', actv2='lrelu', padding_mode=padding_mode)
        
        ## (H/16, W/16) -> (H/8, W/8)
        self.nlam0  = ChengNLAM(in_ch, padding_mode=padding_mode)
        self.block0 = ResBlock(in_ch, main_ch, **normal_block_kwargs)
        self.up0    = UpResBlock(main_ch, main_ch, **up_block_kwargs)

        ## (H/8, W/8) -> (H/4, W/4)
        self.block1 = ResBlock(main_ch, main_ch, **normal_block_kwargs)
        self.up1    = UpResBlock(main_ch, main_ch, **up_block_kwargs)

        ## (H/4, W/4) -> (H/2, W/2)
        self.nlam2  = ChengNLAM(main_ch, padding_mode=padding_mode)
        self.block2 = ResBlock(main_ch, main_ch, **normal_block_kwargs)
        self.up2    = UpResBlock(main_ch, main_ch, **up_block_kwargs)

        ## (H/2, W/2) -> (H, W)
        self.block3 = ResBlock(main_ch, main_ch, **normal_block_kwargs)
        self.up3    = nn.Sequential(
            nn.Conv2d(main_ch, out_ch*4, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.PixelShuffle(2),
        )
        self.use_tanh = use_tanh

    def forward(self, x):
        x = self.nlam0(x)
        x = self.block0(x)
        x = self.up0(x)
        x = self.block1(x)
        x = self.up1(x)
        x = self.nlam2(x)
        x = self.block2(x)
        x = self.up2(x)
        x = self.block3(x)
        x = self.up3(x)
        if self.use_tanh:
            x = torch.tanh(x)
        return x

