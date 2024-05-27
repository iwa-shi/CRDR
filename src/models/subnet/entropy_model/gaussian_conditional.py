from typing import Tuple

from compressai.entropy_models import GaussianConditional as _GaussianConditional
from torch import Tensor

from src.utils.registry import ENTROPYMODEL_REGISTRY


@ENTROPYMODEL_REGISTRY.register()
class GaussianScaleConditional(_GaussianConditional):
    def __init__(self, scale_bound=None):
        super().__init__(scale_table=None, scale_bound=scale_bound)

    def forward(self, y: Tensor, params: Tensor, is_train: bool=True) -> Tuple[Tensor, Tensor]:
        return super().forward(y, scales=params, means=None, training=is_train)

@ENTROPYMODEL_REGISTRY.register()
class GaussianMeanScaleConditional(_GaussianConditional):
    def __init__(self, scale_bound=None):
        super().__init__(scale_table=None, scale_bound=scale_bound)

    def forward(self, y: Tensor, params: Tensor, is_train: bool=True) -> Tuple[Tensor, Tensor]:
        mean, std = params.chunk(2, 1)
        return super().forward(y, scales=std, means=mean, training=is_train)
