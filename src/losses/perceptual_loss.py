from typing import List

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class LPIPSLoss(nn.Module):
    def __init__(self, loss_weight: float, range_norm: bool=False, net: str='vgg'):
        """[summary]

        Args:
            loss_weight (float): [description]
            range_norm (bool, optional): [0, 1] -> [-1, 1]. Defaults to False.
            net (str, optional): [description]. Defaults to 'vgg'.
        """
        super().__init__()
        self.lamb_lpips = loss_weight
        self.range_norm = range_norm
        self.lpips = lpips.LPIPS(net=net)

    def forward(self, real_images, fake_images):
        if self.range_norm:
            real_images = (real_images - 0.5) * 2.
            fake_images = (fake_images - 0.5) * 2.
        lpips_val = self.lpips(real_images, fake_images)
        return self.lamb_lpips * torch.mean(lpips_val)
