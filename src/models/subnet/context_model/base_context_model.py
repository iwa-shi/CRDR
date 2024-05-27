import torch
import torch.nn as nn

class BaseContextModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    
    def forward(self, y_hat) -> torch.Tensor:
        raise NotImplementedError

    def forward_compress(self, y_hat) -> torch.Tensor:
        raise NotImplementedError