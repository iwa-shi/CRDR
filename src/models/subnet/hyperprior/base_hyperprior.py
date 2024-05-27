import torch
import torch.nn as nn

# from src.models.base_module import BaseModule

# class BaseHyperEncoder(BaseModule):
class BaseHyperEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_downscale = None
        self.latent_ch = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# class BaseHyperDecoder(BaseModule):
class BaseHyperDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError