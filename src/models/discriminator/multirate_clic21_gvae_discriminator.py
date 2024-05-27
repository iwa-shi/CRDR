from typing import Optional, Dict, Iterable, List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from src.models.discriminator.base_discriminator import BaseDiscriminator
from src.utils.registry import DISCRIMINATOR_REGISTRY

def conv_norm_lrelu(in_ch, out_ch, kernel_size=3, stride=1, norm_type: str='BN'):
    norm_dict = dict(
        BN=nn.BatchNorm2d,
        IN=nn.InstanceNorm2d,
        none=nn.Identity,
    )
    assert norm_type in norm_dict

    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
        norm_dict[norm_type](out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def build_channel_dict(img_size: int, in_ch: int, main_ch: int, max_ch: int) -> Dict[int, int]:
    """
    ```
    channel_dict = {
        img_size: in_ch,
        img_size//2: main_ch,
        img_size//4: min(main_ch*2, max_ch),
        img_size//8: min(main_ch*4, max_ch), ...
    }
    ```

    Args:
        img_size (int): initial image size ex) 512, 256, 128
        in_ch (int): ex) 3
        main_ch (int): ex) 64, 128
        max_ch (int):

    Returns:
        Dict[int, int]: _description_
    """
    img_size_log2 = int(np.log2(img_size)) # 16: 4, 32: 5, 64: 6, 128: 7, 256: 8
    assert (2 ** img_size_log2) == img_size, f'invalid img_size: {img_size}'
    channel_dict = {img_size: in_ch}
    res = img_size // 2
    ch = main_ch
    for _ in range(img_size_log2-2):
        channel_dict[res] = ch
        ch = min(ch*2, max_ch)
        res //= 2
    return channel_dict


def as_list(val: Union[int, List, Tuple], length: int):
    if isinstance(val, int):
        return [val for _ in range(length)]
    if isinstance(val, (list, tuple)):
        assert len(val) == length, f'len(val) must be {length}, but got {len(val)}'
        return val
    raise TypeError(f'val must be int, list, or tuple, but got {type(val)}')


def rate_ind_to_onehot_feat(image: Tensor, rate_ind: Union[int, float], rate_level: int) -> Tensor:
    N, _, H, W = image.shape
    one_hot = F.one_hot(torch.Tensor([rate_ind]).long(), num_classes=rate_level) # [rate_level]
    one_hot = one_hot.float().to(image.device)
    one_hot = one_hot.reshape(1, -1, 1, 1)
    one_hot = one_hot.repeat(N, 1, H, W)
    return one_hot


class DiscriminatorBlock(nn.Module):
    def __init__(self,
                 channel_dict: Dict,
                 input_res: int,
                 num_depth: int,
                 conv_kwargs: Dict) -> None:
        super().__init__()
        res = input_res
        self.block_resolutions = []
        for i in range(num_depth):
            in_ch = channel_dict[res]
            out_ch = channel_dict[res // 2]
            block = nn.Sequential(
                conv_norm_lrelu(in_ch, out_ch, stride=1, **conv_kwargs),
                conv_norm_lrelu(out_ch, out_ch, stride=2, **conv_kwargs),
            )
            setattr(self, f'b{res}', block)
            self.block_resolutions.append(res)
            res //= 2
        
    def forward(self, x):
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x = block(x)
        return x


class DiscriminatorHead(nn.Module):
    def __init__(self,
                 out_ch: int,
                 channel_dict: Dict,
                 input_res: int,
                 num_depth: int,
                 conv_kwargs: Dict) -> None:
        super().__init__()
        self.block = DiscriminatorBlock(channel_dict, input_res, num_depth, conv_kwargs)
        feat_res = input_res // (2 ** num_depth)
        self.last_conv = nn.Conv2d(channel_dict[feat_res], out_ch, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = self.block(x)
        x = self.last_conv(x)
        return x


@DISCRIMINATOR_REGISTRY.register()
class SharedBackboneClic21GvaeDiscriminator(BaseDiscriminator):
    """
    Shared Backbone + Separate Head
    """
    def __init__(self,
                 num_head: int,
                 in_ch: int=3,
                 out_ch: int=1,
                 main_ch: int=64,
                 img_size: int=256,
                 norm_type: str='none',
                 backbone_depth: int=2,
                 head_depth: int=2,
                 use_rate_ind_cond: bool=False):
        super().__init__()
        channel_dict = build_channel_dict(img_size, in_ch, main_ch, main_ch*8)
        conv_kwargs = dict(kernel_size=3, norm_type=norm_type)
        feat_size = img_size // (2 ** backbone_depth) # backbone output feature map resolution

        self.use_rate_ind_cond = use_rate_ind_cond
        self.rate_level = num_head
        if use_rate_ind_cond:
            channel_dict[img_size] += self.rate_level
        self.backbone = DiscriminatorBlock(channel_dict=channel_dict, input_res=img_size, num_depth=backbone_depth, conv_kwargs=conv_kwargs)

        self.head_list = nn.ModuleList()
        for _ in range(num_head):
            self.head_list.append(
                DiscriminatorHead(out_ch=out_ch, channel_dict=channel_dict, input_res=feat_size, 
                                  num_depth=head_depth, conv_kwargs=conv_kwargs)
            )

    def forward(self, input, rate_ind: Union[float, Tensor], **kwargs):
        if isinstance(rate_ind, torch.Tensor):
            assert rate_ind.numel() == 1
            rate_ind = rate_ind.item()
        rate_ind = int(rate_ind)
        if self.use_rate_ind_cond:
            one_hot = rate_ind_to_onehot_feat(input, rate_ind, self.rate_level)
            input = torch.cat([input, one_hot], dim=1)
        feat = self.backbone(input)
        out = self.head_list[rate_ind](feat)
        return out


@DISCRIMINATOR_REGISTRY.register()
class SharedHeadClic21GvaeDiscriminator(BaseDiscriminator):
    """
    Separate Backbone + Shared Head
    """
    def __init__(self,
                 num_backbone: int,
                 in_ch: int=3,
                 out_ch: int=1,
                 main_ch: int=64,
                 img_size: int=256,
                 norm_type: str='none',
                 backbone_depth: int=2,
                 head_depth: int=2,
                 use_rate_ind_cond: bool=False):
        super().__init__()
        channel_dict = build_channel_dict(img_size, in_ch, main_ch, main_ch*8)
        conv_kwargs = dict(kernel_size=3, norm_type=norm_type)
        feat_size = img_size // (2 ** backbone_depth) # backbone output feature map resolution

        self.backbone_list = nn.ModuleList()
        for _ in range(num_backbone):
            self.backbone_list.append(
                DiscriminatorBlock(channel_dict=channel_dict, input_res=img_size, 
                                   num_depth=backbone_depth, conv_kwargs=conv_kwargs)
            )

        self.use_rate_ind_cond = use_rate_ind_cond
        self.rate_level = num_backbone
        if use_rate_ind_cond:
            channel_dict[feat_size] += self.rate_level

        self.head = DiscriminatorHead(out_ch=out_ch, channel_dict=channel_dict, input_res=feat_size, num_depth=head_depth, conv_kwargs=conv_kwargs)

    def forward(self, input, rate_ind: Union[float, Tensor], **kwargs):
        if isinstance(rate_ind, torch.Tensor):
            assert rate_ind.numel() == 1
            rate_ind = rate_ind.item()
        rate_ind = int(rate_ind)
        feat = self.backbone_list[rate_ind](input)
        if self.use_rate_ind_cond:
            one_hot = rate_ind_to_onehot_feat(feat, rate_ind, self.rate_level)
            feat = torch.cat([feat, one_hot], dim=1)
        out = self.head(feat)
        return out



@DISCRIMINATOR_REGISTRY.register()
class MultirateSeparateClic21GvaeDiscriminator(BaseDiscriminator):
    """
    Separate Discriminator for each rate level
    """
    def __init__(self,
                 rate_level: int,
                 in_ch: int=3,
                 out_ch: int=1,
                 main_ch: Union[int, List[int]]=64,
                 img_size: int=256,
                 norm_type: str='none',
                 depth: Union[int, List[int]]=4):
        super().__init__()
        main_ch_list = as_list(main_ch, length=rate_level)
        depth_list = as_list(depth, length=rate_level)

        ch_dict_list = [build_channel_dict(img_size, in_ch, ch, ch*8) for ch in main_ch_list]

        conv_kwargs = dict(kernel_size=3, norm_type=norm_type)

        self.discriminator_list = nn.ModuleList()

        for i, ch_dict in enumerate(ch_dict_list):
            discriminator = DiscriminatorHead(out_ch, ch_dict, img_size, depth_list[i], conv_kwargs)
            self.discriminator_list.append(discriminator)

    def forward(self, input, rate_ind: Union[float, Tensor], **kwargs):
        if isinstance(rate_ind, torch.Tensor):
            assert rate_ind.numel() == 1
            rate_ind = rate_ind.item()
        rate_ind = int(rate_ind)
        return self.discriminator_list[rate_ind](input)
    

@DISCRIMINATOR_REGISTRY.register()
class MultirateSharedRateCondClic21GvaeDiscriminator(BaseDiscriminator):
    """
    Shared Discriminator with rate_ind Condition
    """
    def __init__(self,
                 rate_level: int,
                 in_ch: int=3,
                 out_ch: int=1,
                 main_ch: int=64,
                 img_size: int=256,
                 norm_type: str='none',
                 depth: int=4,
                 rate_cond_policy: str='onehot'):
        super().__init__()
        assert rate_cond_policy in ['onehot']
        self.rate_cond_policy = rate_cond_policy
        self.rate_level = rate_level

        ch_dict = build_channel_dict(img_size, in_ch+rate_level, main_ch, main_ch*8)

        conv_kwargs = dict(kernel_size=3, norm_type=norm_type)

        self.net = DiscriminatorHead(out_ch, ch_dict, img_size, depth, conv_kwargs)

    def forward(self, input, rate_ind: Union[float, Tensor], **kwargs):
        if isinstance(rate_ind, torch.Tensor):
            assert rate_ind.numel() == 1
            rate_ind = rate_ind.item()
        rate_ind = int(rate_ind)
        one_hot = rate_ind_to_onehot_feat(input, rate_ind, self.rate_level)
        input = torch.cat([input, one_hot], dim=1)
        return self.net(input)

