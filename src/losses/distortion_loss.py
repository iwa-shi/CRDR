from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pytorch_msssim import MS_SSIM

from src.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    def __init__(self,
                 loss_weight: float,
                 normalize_img: bool=True,
                 mse_scale: str='0_1'):
        """_summary_

        Args:
            loss_weight (float): _description_
            normalize_img (bool, optional):
            mse_scale (str, optional): Scale of pixel values for MSE loss
        """
        super().__init__()
        assert normalize_img
        assert mse_scale in ['0_255', '0_1'], f'mse_scale shoukd be "0_255" or "0_1", but {mse_scale} ({type(mse_scale)})'
        self.lamb_mse = loss_weight
        self.mse = nn.MSELoss()
        normalize_func_dict = {'0_255': self.img_range_to_255, '0_1': self.img_range_to_01}
        self.normalize_func = normalize_func_dict[mse_scale]

    @staticmethod
    def img_range_to_255(img: Tensor) -> Tensor:
        img = (img + 1.) / 2. # [-1, 1] -> [0, 1]
        return img * 255. # [0, 1] -> [0, 255]

    @staticmethod
    def img_range_to_01(img: Tensor) -> Tensor:
        return (img + 1.) / 2. # [-1, 1] -> [0, 1]

    def forward(self, real_images: Tensor, fake_images: Tensor, **kwargs):
        if self.normalize_func:
            real_images = self.normalize_func(real_images)
            fake_images = self.normalize_func(fake_images)
        mse = self.mse(real_images, fake_images)
        return self.lamb_mse * mse


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    def __init__(self, loss_weight: float):
        super().__init__()
        self.lamb_l1 = loss_weight
        self.l1_loss = nn.L1Loss()

    def forward(self, real_images, fake_images, **kwargs):
        l1 = self.l1_loss(real_images, fake_images)
        return self.lamb_l1 * l1


@LOSS_REGISTRY.register()
class MSSSIMLoss(nn.Module):
    def __init__(self, loss_weight: float):
        super().__init__()
        self.lamb_msssim = loss_weight
        self.ms_ssim_loss = MS_SSIM(data_range=1, size_average=True, channel=3)

    def forward(self, real_images, fake_images, **kwargs):
        msssim = self.ms_ssim_loss(real_images, fake_images)
        return self.lamb_msssim * (1 - msssim)

