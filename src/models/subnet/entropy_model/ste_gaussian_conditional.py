from typing import Tuple

import torch

from src.models.subnet.entropy_model.gaussian_conditional import GaussianMeanScaleConditional
from src.utils.registry import ENTROPYMODEL_REGISTRY
from .ste_round import ste_round

@ENTROPYMODEL_REGISTRY.register()
class SteGaussianMeanScaleConditional(GaussianMeanScaleConditional):
    """
    Additive uniform noise for entropy calculation, and STE for decoder input
    """
    def __init__(self, scale_bound=None, entropy_quant_type='noise', **kwargs) -> None:
        super().__init__(scale_bound=scale_bound)
        assert entropy_quant_type == 'noise', 'Currently, SteGaussianMeanScaleConditional \
                                                only supports noise quantization for entropy estimation!'
        self.entropy_quant_type = entropy_quant_type

    def forward(self, y: torch.Tensor, params: torch.Tensor, is_train: bool=True) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, _ = params.chunk(2, 1)
        _, y_likelihood = super().forward(y, params, is_train=is_train)
        if is_train:
            y_hat = ste_round(y - mean) + mean
        else:
            y_hat = self.quantize(y, mode='dequantize', means=mean)
        return y_hat, y_likelihood