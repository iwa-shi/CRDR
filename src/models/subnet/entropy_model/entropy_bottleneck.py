from typing import Tuple

import torch
from torch import Tensor

from compressai.entropy_models import EntropyBottleneck as _EntropyBottleneck

from src.utils.registry import ENTROPYMODEL_REGISTRY
from .ste_round import ste_round


@ENTROPYMODEL_REGISTRY.register()
class EntropyBottleneck(_EntropyBottleneck):
    def forward(self, x: Tensor, is_train: bool) -> Tuple[Tensor, Tensor]:
        return super().forward(x, training=is_train)


@ENTROPYMODEL_REGISTRY.register()
class SteEntropyBottleneck(EntropyBottleneck):
    """
    Using noise for entropy calculation, STE for rounded output
    """
    def forward(self, x: Tensor, is_train: bool = True) -> Tuple[Tensor, Tensor]:
        if not is_train:
            return super().forward(x, is_train)
        
        _, x_likelihood = super().forward(x, is_train)
        mu = self._get_medians()
        x_hat = ste_round(x - mu) + mu
        return x_hat, x_likelihood