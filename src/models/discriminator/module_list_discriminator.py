from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor

from src.models.discriminator.base_discriminator import BaseDiscriminator
from src.utils.registry import DISCRIMINATOR_REGISTRY


@DISCRIMINATOR_REGISTRY.register()
class ModuleListDiscriminator(BaseDiscriminator):
    """List of sub-Discriminators
    Each sub-Discriminator has the same architecture but different weights
    """
    def __init__(self, _subd_type, _num_subd, **kwargs) -> None:
        super().__init__()
        self.subD_list = nn.ModuleList()
        for _ in range(_num_subd):
            subD = DISCRIMINATOR_REGISTRY.get(_subd_type)(**kwargs)
            self.subD_list.append(subD)

    def forward(self, input, rate_ind: Union[float, Tensor], **kwargs):
        if isinstance(rate_ind, torch.Tensor):
            assert rate_ind.numel() == 1
            rate_ind = rate_ind.item()
        rate_ind = int(rate_ind)
        return self.subD_list[rate_ind](input, **kwargs)