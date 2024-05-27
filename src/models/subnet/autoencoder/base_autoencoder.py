import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_downscale = None # must be set
        self.latent_ch = None # must be set
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BaseDecoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
